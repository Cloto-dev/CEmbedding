"""Download ONNX embedding models for local inference."""

import argparse
import os
import sys
import urllib.request


def _hf_download(repo_id: str, repo_filename: str, dest_path: str) -> bool:
    """Download a single file from HuggingFace Hub.

    Uses huggingface_hub if available (handles LFS, caching, auth).
    Falls back to direct urllib download for minimal-dependency environments.
    """
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading: {repo_filename} ...")

    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
        import shutil

        shutil.copy2(cached, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")
        return True
    except ImportError:
        pass

    # Fallback: direct URL
    url = f"https://huggingface.co/{repo_id}/resolve/main/{repo_filename}"
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# MiniLM
MINIML_DIR = os.environ.get("ONNX_MODEL_DIR", "data/models/all-MiniLM-L6-v2")
MINIML_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINIML_FILES = {
    "model.onnx": "onnx/model.onnx",
    "tokenizer.json": "tokenizer.json",
}

# jina-v5-nano (retrieval variant with merged LoRA, external data format).
# The repository ships several precisions of the same graph. Each .onnx names
# its external-data file inside the protobuf, so the pair keeps the upstream
# stem on disk; the fp32 pair keeps the historical names "model.onnx" /
# "model.onnx_data" so an existing download stays valid.
JINA_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
JINA_VARIANT_STEMS = {
    "fp32": "model",
    "fp16": "model_fp16",
    "int8": "model_quantized",
}


def jina_v5_nano_files(variant: str = "fp32") -> dict[str, str]:
    """local filename -> repo path for one precision of jina-v5-nano."""
    if variant not in JINA_VARIANT_STEMS:
        raise ValueError(f"unknown jina-v5-nano variant {variant!r}; expected one of {sorted(JINA_VARIANT_STEMS)}")
    stem = JINA_VARIANT_STEMS[variant]
    return {
        f"{stem}.onnx": f"onnx/{stem}.onnx",
        f"{stem}.onnx_data": f"onnx/{stem}.onnx_data",
        "tokenizer.json": "tokenizer.json",
    }


JINA_FILES = jina_v5_nano_files("fp32")

# bge-m3 (Xenova int8 single-file, ~542MB)
# Xenova/bge-m3 is the canonical Transformers.js ONNX conversion maintained by HuggingFace
BGE_M3_REPO = "Xenova/bge-m3"
BGE_M3_FILES = {
    "model.onnx": "onnx/model_int8.onnx",
    "tokenizer.json": "tokenizer.json",
    "sentencepiece.bpe.model": "sentencepiece.bpe.model",
}


def _download_repo_files(repo_id: str, files: dict[str, str], model_dir: str) -> bool:
    """Download a set of repo_filename→local_filename mappings into model_dir."""
    os.makedirs(model_dir, exist_ok=True)
    for local_name, repo_filename in files.items():
        dest = os.path.join(model_dir, local_name)
        if not _hf_download(repo_id, repo_filename, dest):
            return False
    return True


def download():
    """Download MiniLM (legacy entrypoint)."""
    print("=== Downloading all-MiniLM-L6-v2 ONNX model ===")
    ok = _download_repo_files(MINIML_REPO, MINIML_FILES, MINIML_DIR)
    if ok:
        print(f"Model ready at {MINIML_DIR}")
    return ok


def download_jina_v5_nano(model_dir: str = "", variant: str = "fp32") -> bool:
    """Download jina-embeddings-v5-text-nano-retrieval ONNX model (one precision)."""
    if not model_dir:
        model_dir = os.environ.get("ONNX_MODEL_DIR", "data/models/jina-embeddings-v5-text-nano")
    print(f"=== Downloading jina-embeddings-v5-text-nano-retrieval ({variant}) ===")
    ok = _download_repo_files(JINA_REPO, jina_v5_nano_files(variant), model_dir)
    if ok:
        print(f"Model ready at {model_dir}")
    return ok


def download_bge_m3(model_dir: str = "") -> bool:
    """Download BAAI/bge-m3 int8 quantized ONNX model (~542MB) via Xenova conversion."""
    if not model_dir:
        model_dir = os.environ.get("ONNX_MODEL_DIR", "data/models/bge-m3")
    print("=== Downloading BAAI/bge-m3 ONNX int8 (~542 MB) from Xenova/bge-m3 ===")
    ok = _download_repo_files(BGE_M3_REPO, BGE_M3_FILES, model_dir)
    if ok:
        print(f"Model ready at {model_dir}")
    return ok


def main():
    """Console-script / ``python -m cembedding.download_model`` entry point."""
    parser = argparse.ArgumentParser(description="Download ONNX embedding models")
    parser.add_argument("--model", default="miniml", choices=["miniml", "jina-v5-nano", "bge-m3"])
    parser.add_argument(
        "--variant",
        default="fp32",
        choices=sorted(JINA_VARIANT_STEMS),
        help="precision of jina-v5-nano to fetch (ignored for other models); the server "
        "selects it at runtime with EMBEDDING_MODEL_VARIANT",
    )
    args = parser.parse_args()

    if args.model == "miniml":
        success = download()
    elif args.model == "jina-v5-nano":
        success = download_jina_v5_nano(variant=args.variant)
    else:
        success = download_bge_m3()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
