"""
Continued Pre-Training (CPT) of Qwen3-0.6B on financial news data.

Trains with the standard causal-LM objective (next-token prediction) on chunked
text, then compares perplexity on the held-out test set before and after.

Usage (xpu-test conda env):
    python train_cpt.py \\
        --train-samples 2000 \\
        --epochs 1 \\
        --device xpu

Full run on news_training.jsonl:
    python train_cpt.py --train-samples 0 --epochs 1 --device xpu
    (0 = use all 46 K documents)
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()
LOG = logging.getLogger(__name__)

DEFAULT_MODEL      = "Qwen/Qwen3-0.6B"
DEFAULT_TRAIN_DATA = "data/raw/news_bulk/news_training.jsonl"
DEFAULT_EVAL_DATA  = "data/raw/news_bulk/news_test.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/qwen3-0.6b-finance"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def sample_jsonl(path: Path, n: int, seed: int = 42) -> list[str]:
    """Read texts from JSONL, optionally sub-sample n rows."""
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                t = json.loads(line).get("text", "")
                if len(t.split()) >= 20:
                    texts.append(t)
            except Exception:
                pass
    if n and n < len(texts):
        random.seed(seed)
        texts = random.sample(texts, n)
    LOG.info("Loaded %d training texts from %s", len(texts), path)
    return texts


def make_chunks(texts: list[str], tokenizer, max_length: int = 512) -> list[torch.Tensor]:
    """
    Concatenate all texts separated by EOS, then split into fixed-length chunks.
    This wastes no tokens and is the standard CPT data preparation approach.
    """
    eos = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        all_ids.extend(ids)
        all_ids.append(eos)

    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        chunks.append(torch.tensor(all_ids[i : i + max_length], dtype=torch.long))

    LOG.info(
        "Tokenised %d texts → %d tokens → %d chunks of %d",
        len(texts), len(all_ids), len(chunks), max_length,
    )
    return chunks


class ChunkDataset(Dataset):
    def __init__(self, chunks: list[torch.Tensor]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        ids = self.chunks[i]
        return {"input_ids": ids, "labels": ids}


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def compute_ppl(
    model,
    tokenizer,
    eval_texts: list[str],
    device: str,
    max_tokens: int = 256,
    stride: int = 128,
) -> float:
    """Sliding-window causal-LM perplexity (same method as eval_perplexity.py)."""
    model.eval()
    nlls, skipped = [], 0
    with torch.no_grad():
        for text in eval_texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_tokens
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.size(1) < 4:
                skipped += 1
                continue
            seq_len   = input_ids.size(1)
            total_nll = 0.0
            total_tok = 0
            prev_end  = 0
            for begin in range(0, seq_len, stride):
                end        = min(begin + max_tokens, seq_len)
                target_len = end - prev_end
                chunk      = input_ids[:, begin:end]
                labels     = chunk.clone()
                labels[:, :-target_len] = -100
                loss = model(chunk, labels=labels).loss
                total_nll += loss.item() * target_len
                total_tok += target_len
                prev_end   = end
                if end >= seq_len:
                    break
            if total_tok > 0 and math.isfinite(total_nll):
                nlls.append(total_nll / total_tok)
            else:
                skipped += 1

    model.train()
    if not nlls:
        return float("nan")
    mean_nll = sum(nlls) / len(nlls)
    LOG.info("PPL eval: %d docs, %d skipped → PPL %.2f", len(nlls), skipped, math.exp(mean_nll))
    return math.exp(mean_nll)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model,
    dataset: ChunkDataset,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    warmup_steps: int,
    log_every: int = 20,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,   # XPU does not use CUDA pinned memory
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / grad_accum) * epochs
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    is_xpu   = device.startswith("xpu")
    is_cuda  = device.startswith("cuda")
    use_amp  = is_xpu or is_cuda
    amp_dtype = torch.bfloat16

    LOG.info(
        "Training: %d chunks, batch=%d, grad_accum=%d, effective_batch=%d, "
        "steps=%d, lr=%.0e, warmup=%d, amp=%s",
        len(dataset), batch_size, grad_accum, batch_size * grad_accum,
        total_steps, lr, warmup_steps, use_amp,
    )

    global_step = 0
    running_loss = 0.0
    t0 = time.monotonic()

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            if use_amp:
                dev_type = "xpu" if is_xpu else "cuda"
                with torch.autocast(device_type=dev_type, dtype=amp_dtype):
                    loss = model(input_ids=input_ids, labels=labels).loss
            else:
                loss = model(input_ids=input_ids, labels=labels).loss

            (loss / grad_accum).backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0 or global_step == 1:
                    elapsed = time.monotonic() - t0
                    avg_loss = running_loss / (log_every * grad_accum)
                    steps_left = total_steps - global_step
                    eta = elapsed / global_step * steps_left if global_step else 0
                    LOG.info(
                        "epoch %d/%d  step %d/%d  loss=%.4f  lr=%.2e  "
                        "elapsed=%.0fs  ETA=%.0fs",
                        epoch + 1, epochs, global_step, total_steps,
                        avg_loss, scheduler.get_last_lr()[0],
                        elapsed, eta,
                    )
                    running_loss = 0.0

    LOG.info("Training done in %.0fs", time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    return requested


def main():
    parser = argparse.ArgumentParser(description="CPT of Qwen3-0.6B on financial news")
    parser.add_argument("--model",         default=DEFAULT_MODEL)
    parser.add_argument("--train-data",    default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data",     default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-dir",    default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-samples", type=int, default=2000,
                        help="Docs to sample from train file (0 = all)")
    parser.add_argument("--eval-samples",  type=int, default=200,
                        help="Docs for PPL eval before/after")
    parser.add_argument("--chunk-size",    type=int, default=512)
    parser.add_argument("--epochs",        type=int, default=1)
    parser.add_argument("--batch-size",    type=int, default=4)
    parser.add_argument("--grad-accum",    type=int, default=8)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--warmup-steps",  type=int, default=50)
    parser.add_argument("--log-every",     type=int, default=10)
    parser.add_argument("--device",        default="auto")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/train_cpt.log"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    random.seed(args.seed)
    device = resolve_device(args.device)
    LOG.info("Device: %s", device)
    if device.startswith("xpu"):
        LOG.info("XPU: %s  (%d MB)", torch.xpu.get_device_name(0),
                 torch.xpu.get_device_properties(0).total_memory // 1024**2)

    # ---- tokenizer ----
    LOG.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- load model ----
    LOG.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    LOG.info("Model loaded: %.0fM params on %s", n_params, device)

    # ---- eval texts (loaded once, reused before + after) ----
    eval_texts = []
    with open(args.eval_data, encoding="utf-8") as fh:
        for line in fh:
            if len(eval_texts) >= args.eval_samples:
                break
            try:
                t = json.loads(line).get("text", "")
                if len(t.split()) >= 20:
                    eval_texts.append(t)
            except Exception:
                pass
    LOG.info("Loaded %d eval texts from %s", len(eval_texts), args.eval_data)

    # ---- PPL before training ----
    LOG.info("=== PPL BEFORE TRAINING ===")
    ppl_before = compute_ppl(model, tokenizer, eval_texts, device,
                             max_tokens=256, stride=128)
    LOG.info("PPL before: %.2f", ppl_before)

    # ---- training data ----
    train_texts = sample_jsonl(Path(args.train_data), args.train_samples, args.seed)
    chunks      = make_chunks(train_texts, tokenizer, args.chunk_size)
    dataset     = ChunkDataset(chunks)

    if len(chunks) == 0:
        LOG.error("No training chunks produced — try increasing --train-samples.")
        return

    # ---- train ----
    LOG.info("=== TRAINING ===")
    train(
        model, dataset, device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
    )

    # ---- save ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Saving model to %s ...", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    LOG.info("Model saved.")

    # ---- PPL after training ----
    LOG.info("=== PPL AFTER TRAINING ===")
    ppl_after = compute_ppl(model, tokenizer, eval_texts, device,
                            max_tokens=256, stride=128)
    LOG.info("PPL after: %.2f", ppl_after)

    # ---- summary ----
    delta = ppl_before - ppl_after
    pct   = delta / ppl_before * 100
    print("\n" + "=" * 60)
    print("  CONTINUED PRE-TRAINING RESULTS")
    print("=" * 60)
    print(f"  Model          : {args.model}")
    print(f"  Device         : {device}")
    print(f"  Train docs     : {len(train_texts):,}  ({len(chunks):,} chunks of {args.chunk_size} tokens)")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Eval docs      : {len(eval_texts)}")
    print(f"  PPL before     : {ppl_before:.2f}")
    print(f"  PPL after      : {ppl_after:.2f}")
    print(f"  Improvement    : {delta:+.2f}  ({pct:+.1f}%)")
    print(f"  Saved to       : {out_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
