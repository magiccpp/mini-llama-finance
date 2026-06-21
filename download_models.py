"""
Download HuggingFace models used by this project.

Skips models that are already fully cached.
Gated models (Llama) require HF_TOKEN env var or `huggingface-cli login`.

Usage
-----
# Download specific models
python download_models.py --models qwen3-0.6b,qwen3-1.7b,qwen3-4b

# Download everything
python download_models.py --models all

# List available model keys and their cache status
python download_models.py --list
"""

import argparse
import logging
import os
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry — add new models here
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "llama-8b":   {
        "id":    "meta-llama/Meta-Llama-3.1-8B",
        "gated": True,
        "note":  "Requires HF_TOKEN (Meta gated model)",
    },
    "qwen3-0.6b": {"id": "Qwen/Qwen3-0.6B",  "gated": False, "note": ""},
    "qwen3-1.7b": {"id": "Qwen/Qwen3-1.7B",  "gated": False, "note": ""},
    "qwen3-4b":   {"id": "Qwen/Qwen3-4B",    "gated": False, "note": ""},
    "qwen3-8b":   {"id": "Qwen/Qwen3-8B",    "gated": False, "note": ""},
}

# Patterns to skip — saves ~30-50% bandwidth by skipping TF/Flax/ONNX weights
IGNORE_PATTERNS = [
    "*.msgpack",       # Flax
    "*.h5",            # TensorFlow
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "*.ot",            # OpenVINO
    "onnx/*",
]

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------

def _cache_dir(model_id: str) -> Path:
    """Return the expected HF cache directory for a model."""
    return HF_CACHE / f"models--{model_id.replace('/', '--')}"


def is_cached(model_id: str) -> bool:
    """
    Return True if the model appears to be fully downloaded.
    Checks that the cache directory exists and contains at least one
    non-empty snapshot directory.
    """
    snapshots = _cache_dir(model_id) / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir():
            files = list(snap.iterdir())
            # A real snapshot has config.json + weight files
            if any(f.name == "config.json" for f in files):
                return True
    return False


def cache_size_gb(model_id: str) -> float:
    """Rough size of the cached model in GB."""
    d = _cache_dir(model_id)
    if not d.exists():
        return 0.0
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return total / 1e9


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_model(key: str, entry: dict, hf_token: str | None) -> bool:
    """Download one model. Returns True on success."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    model_id = entry["id"]
    gated    = entry["gated"]

    if gated and not hf_token:
        LOG.error(
            "%-12s  SKIPPED — gated model requires HF_TOKEN env var "
            "or `huggingface-cli login`",
            key,
        )
        return False

    LOG.info("%-12s  Downloading %s ...", key, model_id)
    t0 = __import__("time").monotonic()
    try:
        path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            ignore_patterns=IGNORE_PATTERNS,
        )
        elapsed = __import__("time").monotonic() - t0
        size    = cache_size_gb(model_id)
        LOG.info("%-12s  Done  (%.1f GB, %.0fs)  →  %s", key, size, elapsed, path)
        return True
    except GatedRepoError:
        LOG.error(
            "%-12s  Access denied — request access at "
            "https://huggingface.co/%s then re-run.",
            key, model_id,
        )
        return False
    except RepositoryNotFoundError:
        LOG.error("%-12s  Repository not found: %s", key, model_id)
        return False
    except Exception as exc:
        LOG.error("%-12s  Failed: %s", key, exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_models():
    print(f"\n{'Key':<14} {'HF model ID':<35} {'Cached':<8} {'Size':>6}  Notes")
    print("-" * 78)
    for key, entry in MODEL_REGISTRY.items():
        cached = is_cached(entry["id"])
        size   = f"{cache_size_gb(entry['id']):.1f}G" if cached else "-"
        status = "yes" if cached else "no"
        note   = entry["note"] or ("-" if not entry["gated"] else "gated")
        print(f"{key:<14} {entry['id']:<35} {status:<8} {size:>6}  {note}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for this project",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    keys_str = ", ".join(MODEL_REGISTRY)
    parser.add_argument(
        "--models", default="",
        help=f"Comma-separated model keys to download, or 'all'.\nAvailable: {keys_str}",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Show all models and their cache status, then exit",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if already cached",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list or not args.models:
        list_models()
        if not args.models:
            parser.print_help()
        return

    # Resolve model keys
    if args.models.strip().lower() == "all":
        requested = list(MODEL_REGISTRY.keys())
    else:
        requested = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    unknown = [k for k in requested if k not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown model key(s): {unknown}")
        print(f"Available: {keys_str}")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    results = {"downloaded": [], "skipped": [], "failed": []}

    for key in requested:
        entry    = MODEL_REGISTRY[key]
        model_id = entry["id"]

        if not args.force and is_cached(model_id):
            size = cache_size_gb(model_id)
            LOG.info("%-12s  Already cached (%.1f GB) — skipping.  Use --force to re-download.", key, size)
            results["skipped"].append(key)
            continue

        ok = download_model(key, entry, hf_token)
        results["downloaded" if ok else "failed"].append(key)

    # Summary
    print("\n" + "=" * 50)
    print("  DOWNLOAD SUMMARY")
    print("=" * 50)
    if results["downloaded"]:
        print(f"  Downloaded : {', '.join(results['downloaded'])}")
    if results["skipped"]:
        print(f"  Skipped    : {', '.join(results['skipped'])}  (already cached)")
    if results["failed"]:
        print(f"  Failed     : {', '.join(results['failed'])}")
    print("=" * 50 + "\n")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
