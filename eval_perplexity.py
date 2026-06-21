"""
Perplexity evaluation on financial news data.

Models
------
1. ProsusAI/finbert  (BERT masked LM → pseudo-perplexity via masked-token scoring)
2. meta-llama/Meta-Llama-3.1-8B  (causal LM → standard sliding-window perplexity)

Notes
-----
• FinBERT is a BERT classification model; we load it as BertForMaskedLM so the
  BERT encoder (and its tied embedding matrix) is used for pseudo-perplexity.
  The MLM projection transform will be randomly initialised — the metric is an
  *approximate* pseudo-perplexity of the encoder, not exact MLM perplexity.
• Llama 3.1 8B is a gated model on HuggingFace. Set HF_TOKEN env var or run
  `huggingface-cli login` first.
• Intel Arc GPU (XPU) requires intel_extension_for_pytorch:
    pip install intel_extension_for_pytorch \\
        --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  Level Zero runtime must also be installed (intel-level-zero-gpu package).

Devices
-------
  auto   — CUDA → XPU → CPU (in order of preference)
  cuda   — NVIDIA GPU
  xpu    — Intel Arc / Xe GPU via oneAPI Level Zero
  cpu    — host CPU (slowest; always available)

Usage
-----
python eval_perplexity.py \\
    --data   data/raw/news_bulk/news_bulk_test.jsonl \\
    --models finbert,llama \\
    --finbert-samples 200 \\
    --llama-samples  20  \\
    --max-tokens     256 \\
    --device xpu
"""

import argparse
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()   # silence transformer weight warnings
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intel XPU / IPEX helpers
# ---------------------------------------------------------------------------

def _try_import_ipex():
    """Return the ipex module if available and functional, else None."""
    try:
        import intel_extension_for_pytorch as ipex
        return ipex
    except (ImportError, OSError):
        return None


def _xpu_available() -> bool:
    if not hasattr(torch, "xpu"):
        return False
    _try_import_ipex()   # side-effect: registers XPU backend
    return torch.xpu.is_available()


def _ipex_optimize_model(model, is_llm: bool = False, dtype=torch.bfloat16):
    """
    Apply IPEX graph optimisation for Intel GPU.
    Uses optimize_transformers() for LLMs (better kernel fusion),
    optimize() for smaller encoder models.
    Falls back silently if the IPEX/PyTorch build has API mismatches.
    """
    ipex = _try_import_ipex()
    if ipex is None:
        return model
    try:
        if is_llm and hasattr(ipex, "optimize_transformers"):
            return ipex.optimize_transformers(model, device="xpu", dtype=dtype, inplace=True)
        return ipex.optimize(model, dtype=dtype, inplace=True)
    except (AttributeError, RuntimeError) as e:
        LOG.warning("IPEX optimisation skipped (%s: %s); model runs on XPU without it.", type(e).__name__, e)
        return model

# Model registry — add new models here; the rest of the script picks them up.
# type: "masked_lm"  → pseudo-perplexity via per-token masking (BERT-style)
#       "causal_lm"  → standard sliding-window NLL perplexity (GPT-style)
# gated: True means a HuggingFace token (HF_TOKEN env var) is required.
MODEL_REGISTRY: dict[str, dict] = {
    # --- BERT-style encoder ---
    "finbert":    {"id": "ProsusAI/finbert",             "type": "masked_lm", "gated": False},
    # --- Llama ---
    "llama-8b":   {"id": "meta-llama/Meta-Llama-3.1-8B", "type": "causal_lm", "gated": True},
    # --- Qwen3  (no gate; sizes closest to 0.8 B / 2 B / 4 B / 8 B) ---
    "qwen3-0.6b": {"id": "Qwen/Qwen3-0.6B",             "type": "causal_lm", "gated": False},
    "qwen3-1.7b": {"id": "Qwen/Qwen3-1.7B",             "type": "causal_lm", "gated": False},
    "qwen3-4b":   {"id": "Qwen/Qwen3-4B",               "type": "causal_lm", "gated": False},
    "qwen3-8b":   {"id": "Qwen/Qwen3-8B",               "type": "causal_lm", "gated": False},
}

# Keep these as convenient aliases for the old CLI flags
FINBERT_MODEL = MODEL_REGISTRY["finbert"]["id"]
LLAMA_MODEL   = MODEL_REGISTRY["llama-8b"]["id"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts(path: Path, max_samples: int, min_words: int = 30) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if len(texts) >= max_samples:
                break
            try:
                text = json.loads(line).get("text", "")
                if len(text.split()) >= min_words:
                    texts.append(text)
            except Exception:
                pass
    LOG.info("Loaded %d texts from %s", len(texts), path)
    return texts


# ---------------------------------------------------------------------------
# FinBERT — pseudo-perplexity (masked-token scoring)
# ---------------------------------------------------------------------------

def _pseudo_ppl_one(model, input_ids: torch.Tensor, mask_token_id: int) -> float | None:
    """
    Compute pseudo-perplexity for one tokenised example.
    Creates a batch with one masked copy per non-special token,
    runs a single forward pass, then reads off the diagonal log-probs.
    Returns the per-token NLL (not the exponentiated PPL).
    """
    seq_len = input_ids.size(1)
    # Positions to score: skip [CLS]=0 and [SEP]=seq_len-1
    positions = list(range(1, seq_len - 1))
    if not positions:
        return None

    # Build batch: each row has exactly one token masked
    batch = input_ids.expand(len(positions), -1).clone()   # [n_pos, seq_len]
    for i, pos in enumerate(positions):
        batch[i, pos] = mask_token_id

    with torch.no_grad():
        logits = model(input_ids=batch).logits   # [n_pos, seq_len, vocab]

    log_probs = torch.log_softmax(logits, dim=-1)   # [n_pos, seq_len, vocab]
    original   = input_ids[0]                        # [seq_len]

    nll_sum = 0.0
    for i, pos in enumerate(positions):
        true_token = original[pos].item()
        nll_sum -= log_probs[i, pos, true_token].item()

    return nll_sum / len(positions)   # mean NLL per token


def compute_finbert_pseudo_ppl(
    texts: list[str],
    model_name: str = FINBERT_MODEL,
    max_tokens: int = 128,
    device: str = "cpu",
) -> dict:
    LOG.info("Loading FinBERT: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load_as_mlm=True loads BertForMaskedLM; the MLM projection transform
    # is randomly initialised (ProsusAI/finbert is a classifier checkpoint)
    # but the encoder + embedding weights ARE loaded from the checkpoint.
    # For XPU: load on CPU first, then move — avoids device_map XPU edge cases.
    load_device = "cpu" if device.startswith("xpu") else device
    dtype = torch.bfloat16 if device.startswith("xpu") else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    ).to(load_device).eval()

    if device.startswith("xpu"):
        model = model.to(device)
        model = _ipex_optimize_model(model, is_llm=False, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    LOG.info("FinBERT loaded (%.0fM params) on %s", n_params, device)

    nlls, skipped = [], 0
    t0 = time.monotonic()

    for text in tqdm(texts, desc="FinBERT pseudo-PPL", unit="doc"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.size(1) < 4:   # need at least 1 scorable token
            skipped += 1
            continue
        nll = _pseudo_ppl_one(model, input_ids, tokenizer.mask_token_id)
        if nll is not None and math.isfinite(nll):
            nlls.append(nll)
        else:
            skipped += 1

    elapsed = time.monotonic() - t0

    if not nlls:
        return {"model": model_name, "error": "No valid samples"}

    mean_nll = statistics.mean(nlls)
    return {
        "model":      model_name,
        "type":       "pseudo-perplexity (masked LM)",
        "device":     device,
        "samples":    len(nlls),
        "skipped":    skipped,
        "ppl_mean":   round(math.exp(mean_nll), 2),
        "ppl_median": round(math.exp(statistics.median(nlls)), 2),
        "ppl_min":    round(math.exp(min(nlls)), 2),
        "ppl_max":    round(math.exp(max(nlls)), 2),
        "elapsed_s":  round(elapsed, 1),
        "note": (
            "MLM projection transform is randomly initialised "
            "(ProsusAI/finbert is a classifier checkpoint). "
            "Encoder + embedding weights are loaded from FinBERT."
        ),
    }


# ---------------------------------------------------------------------------
# Llama 3.1 8B — standard causal-LM perplexity (sliding window)
# ---------------------------------------------------------------------------

def _causal_ppl_one(
    model,
    input_ids: torch.Tensor,
    max_tokens: int,
    stride: int,
) -> tuple[float, int]:
    """
    Compute average NLL using the sliding-window method
    (Equation 4 of https://arxiv.org/abs/2109.01652).
    Returns (total_nll, total_tokens_scored).
    """
    seq_len   = input_ids.size(1)
    total_nll = 0.0
    total_tok = 0
    prev_end  = 0

    for begin in range(0, seq_len, stride):
        end       = min(begin + max_tokens, seq_len)
        # Only score the tokens that are new in this window
        target_len = end - prev_end
        chunk      = input_ids[:, begin:end]
        # Mask out the overlap so loss ignores already-scored tokens
        labels = chunk.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            loss = model(chunk, labels=labels).loss   # mean NLL over target tokens
        total_nll += loss.item() * target_len
        total_tok += target_len
        prev_end   = end
        if end >= seq_len:
            break

    return total_nll, total_tok


def compute_llama_ppl(
    texts: list[str],
    model_name: str = LLAMA_MODEL,
    max_tokens: int  = 512,
    stride: int      = 256,
    device: str      = "cpu",
    load_in_4bit: bool = False,
) -> dict:
    LOG.info("Loading Llama: %s (device=%s, 4bit=%s)", model_name, device, load_in_4bit)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    is_xpu   = device.startswith("xpu")

    # XPU: load on CPU with low_cpu_mem_usage, then move to XPU.
    # Using device_map="xpu" directly is not reliably supported in all
    # transformers versions; CPU→XPU is safer.
    if is_xpu:
        load_kwargs = dict(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
    else:
        load_kwargs = dict(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=hf_token,
        )

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs.pop("torch_dtype", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model     = AutoModelForCausalLM.from_pretrained(**load_kwargs).eval()

    if is_xpu:
        LOG.info("Moving model to %s ...", device)
        model = model.to(device)
        model = _ipex_optimize_model(model, is_llm=True, dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    LOG.info("Llama loaded (%.0fM params) on %s", n_params, device)

    nlls, skipped = [], 0
    t0 = time.monotonic()

    for text in tqdm(texts, desc="Llama PPL", unit="doc"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        input_ids = enc["input_ids"].to(device)
        if input_ids.size(1) < 4:
            skipped += 1
            continue
        try:
            total_nll, total_tok = _causal_ppl_one(model, input_ids, max_tokens, stride)
            if total_tok > 0 and math.isfinite(total_nll):
                nlls.append(total_nll / total_tok)
        except Exception as exc:
            LOG.warning("Sample failed: %s", exc)
            skipped += 1

    elapsed = time.monotonic() - t0

    if not nlls:
        return {"model": model_name, "error": "No valid samples"}

    mean_nll = statistics.mean(nlls)
    return {
        "model":      model_name,
        "type":       "perplexity (causal LM, sliding window)",
        "device":     device,
        "samples":    len(nlls),
        "skipped":    skipped,
        "ppl_mean":   round(math.exp(mean_nll), 2),
        "ppl_median": round(math.exp(statistics.median(nlls)), 2),
        "ppl_min":    round(math.exp(min(nlls)), 2),
        "ppl_max":    round(math.exp(max(nlls)), 2),
        "elapsed_s":  round(elapsed, 1),
        "stride":     stride,
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def print_results(results: list[dict]):
    print("\n" + "=" * 70)
    print("  PERPLEXITY RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\nModel : {r['model']}")
        print(f"Type  : {r.get('type', 'N/A')}")
        if "error" in r:
            print(f"ERROR : {r['error']}")
            continue
        print(f"Docs  : {r['samples']} evaluated, {r.get('skipped',0)} skipped")
        print(f"PPL   : mean={r['ppl_mean']:>9.2f}  median={r['ppl_median']:>9.2f}"
              f"  min={r['ppl_min']:>9.2f}  max={r['ppl_max']:>9.2f}")
        if "device" in r:
            print(f"Device: {r['device']}")
        if "elapsed_s" in r:
            secs = r["elapsed_s"]
            per_doc = secs / r["samples"] if r["samples"] else 0
            print(f"Time  : {secs:.1f}s total  ({per_doc:.2f}s / doc)")
        if "note" in r:
            # Wrap note at 65 chars
            words = r["note"].split()
            line, lines = "", []
            for w in words:
                if len(line) + len(w) + 1 > 65:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            print("Note  : " + ("\n        ".join(lines)))
    print("\n" + "=" * 70)

    # Comparison table
    valid = [r for r in results if "ppl_mean" in r]
    if len(valid) >= 2:
        print("\n  COMPARISON  (lower PPL = better fit to financial text)")
        print(f"  {'Model key':<14} {'HF model':<32} {'Device':<6} {'PPL mean':>10}  {'PPL median':>10}  {'Time/doc':>9}")
        print("  " + "-" * 85)
        for r in sorted(valid, key=lambda x: x["ppl_mean"]):
            key     = r.get("key", r["model"].split("/")[-1])
            hf_id   = r["model"].split("/")[-1]
            dev     = r.get("device", "?")
            per_doc = r["elapsed_s"] / r["samples"] if r.get("samples") else 0
            print(f"  {key:<14} {hf_id:<32} {dev:<6} {r['ppl_mean']:>10.2f}  {r['ppl_median']:>10.2f}  {per_doc:>7.2f}s")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    model_keys = ", ".join(MODEL_REGISTRY)
    parser = argparse.ArgumentParser(
        description="Perplexity eval for financial news data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data", default="data/raw/news_bulk/news_bulk_test.jsonl",
                        help="Path to JSONL test file")
    parser.add_argument("--models", default="finbert,llama-8b",
                        help=f"Comma-separated model keys (default: finbert,llama-8b)\n"
                             f"Available: {model_keys}")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of docs to evaluate per model (default 100)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per document (default 256)")
    parser.add_argument("--stride", type=int, default=128,
                        help="Sliding-window stride for causal LMs (default 128)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load causal LMs in 4-bit quantisation (requires CUDA + bitsandbytes)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto | cpu | cuda | xpu (default: auto → CUDA→XPU→CPU)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Resolve device: CUDA → XPU → CPU
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif _xpu_available():
            device = "xpu"
        else:
            device = "cpu"
    else:
        device = args.device

    # Validate XPU request
    if device.startswith("xpu"):
        if not _xpu_available():
            LOG.error(
                "XPU requested but not available.\n"
                "  Required: PyTorch XPU build + intel_extension_for_pytorch + Level Zero.\n"
                "  Use the xpu-test conda env:\n"
                "    /home/ken/anaconda3/envs/xpu-test/bin/python eval_perplexity.py --device xpu\n"
                "  Falling back to CPU."
            )
            device = "cpu"
        else:
            xpu_name = torch.xpu.get_device_name(0)
            xpu_mem  = torch.xpu.get_device_properties(0).total_memory // 1024**2
            LOG.info("XPU device: %s  (%d MB)", xpu_name, xpu_mem)

    LOG.info("Using device: %s", device)
    if device == "cpu":
        LOG.warning(
            "Running on CPU — causal LMs will be slow (~1–3 tok/s).\n"
            "  Tip: use --device xpu with the xpu-test conda env for Intel Arc acceleration."
        )

    # Parse and validate requested models
    requested = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in MODEL_REGISTRY]
    if unknown:
        parser.error(f"Unknown model key(s): {unknown}\nAvailable: {model_keys}")

    hf_token  = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    data_path = Path(args.data)
    texts_cache: dict[int, list[str]] = {}   # sample-count → texts (avoid re-reading)
    results: list[dict] = []

    for key in requested:
        entry      = MODEL_REGISTRY[key]
        model_id   = entry["id"]
        model_type = entry["type"]
        gated      = entry["gated"]

        if gated and not hf_token:
            LOG.warning(
                "Model %s is gated — set HF_TOKEN env var or run `huggingface-cli login`.", key
            )

        if args.samples not in texts_cache:
            texts_cache[args.samples] = load_texts(data_path, args.samples)
        texts = texts_cache[args.samples]

        LOG.info("--- %s (%s, %s) ---", key, model_id, model_type)

        if model_type == "masked_lm":
            res = compute_finbert_pseudo_ppl(
                texts, model_name=model_id, max_tokens=args.max_tokens, device=device,
            )
        else:  # causal_lm
            res = compute_llama_ppl(
                texts, model_name=model_id, max_tokens=args.max_tokens,
                stride=args.stride, device=device, load_in_4bit=args.load_in_4bit,
            )

        res["key"] = key   # short name for comparison table
        results.append(res)
        LOG.info("%s done: PPL mean=%.2f", key, res.get("ppl_mean", float("nan")))

    print_results(results)

    out = data_path.parent / "perplexity_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    LOG.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
