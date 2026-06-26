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

# jina-v5-nano (retrieval variant with merged LoRA, external data format)
JINA_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
JINA_FILES = {
    "model.onnx": "onnx/model.onnx",
    "model.onnx_data": "onnx/model.onnx_data",
    "tokenizer.json": "tokenizer.json",
}

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


def download_jina_v5_nano(model_dir: str = "") -> bool:
    """Download jina-embeddings-v5-text-nano-retrieval ONNX model."""
    if not model_dir:
        model_dir = os.environ.get("ONNX_MODEL_DIR", "data/models/jina-embeddings-v5-text-nano")
    print("=== Downloading jina-embeddings-v5-text-nano-retrieval ===")
    ok = _download_repo_files(JINA_REPO, JINA_FILES, model_dir)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ONNX embedding models")
    parser.add_argument("--model", default="miniml", choices=["miniml", "jina-v5-nano", "bge-m3"])
    args = parser.parse_args()

    if args.model == "miniml":
        success = download()
    elif args.model == "jina-v5-nano":
        success = download_jina_v5_nano()
    else:
        success = download_bge_m3()
    sys.exit(0 if success else 1)
