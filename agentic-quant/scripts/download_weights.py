# =============================================================================
# AGENTIC-QUANT — Download Model Weights
#
# Script download model weights tu HuggingFace Hub hoac URL fallback.
# Ho tro LSTM, XGBoost Model A, XGBoost Model B.
#
# Usage:
#   python scripts/download_weights.py --model-type all --output-dir ./models
#   python scripts/download_weights.py --model-type lstm
#   python scripts/download_weights.py --model-type xgb-a --output-dir /tmp/models
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger


# =============================================================================
# Model URLs
# =============================================================================

MODEL_URLS: dict[str, str] = {
    "lstm": "https://huggingface.co/agentic-quant/lstm-ae/resolve/main/lstm_ae.pt",
    "xgb-a": "https://huggingface.co/agentic-quant/xgboost/resolve/main/model_a.json",
    "xgb-b": "https://huggingface.co/agentic-quant/xgboost/resolve/main/model_b.json",
}

# Checksums (SHA256) — neu co, verify sau khi download
MODEL_CHECKSUMS: dict[str, str | None] = {
    "lstm": None,   # TODO: Add checksum when available
    "xgb-a": None,
    "xgb-b": None,
}

MODEL_NAMES: dict[str, str] = {
    "lstm": "LSTM Autoencoder",
    "xgb-a": "XGBoost Model A (Direction)",
    "xgb-b": "XGBoost Model B (Zone Hold)",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "lstm": "LSTM autoencoder weights (PyTorch .pt)",
    "xgb-a": "XGBoost model A — direction prediction (BSL/SSL/LATERAL)",
    "xgb-b": "XGBoost model B — zone hold prediction",
}


# =============================================================================
# Helpers
# =============================================================================


def _compute_sha256(filepath: Path) -> str:
    """Tinh SHA256 checksum cua file.

    Args:
        filepath: Path to file.

    Returns:
        SHA256 hex digest.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _download_with_hf_hub(
    model_type: str,
    output_path: Path,
) -> bool:
    """Download model weights tu HuggingFace Hub.

    Args:
        model_type: Loai model (lstm, xgb-a, xgb-b).
        output_path: Path luu file output.

    Returns:
        True neu download thanh cong, False neu that bai.
    """
    try:
        from huggingface_hub import hf_hub_download

        # Map model type to HF repo info
        hf_repos: dict[str, tuple[str, str]] = {
            "lstm": ("agentic-quant/lstm-ae", "lstm_ae.pt"),
            "xgb-a": ("agentic-quant/xgboost", "model_a.json"),
            "xgb-b": ("agentic-quant/xgboost", "model_b.json"),
        }

        if model_type not in hf_repos:
            logger.error(f"Unknown model type: {model_type}")
            return False

        repo_id, filename = hf_repos[model_type]
        logger.info(f"Downloading {model_type} from HF Hub: {repo_id}/{filename}")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_path.parent,
            local_dir_use_symlinks=False,
        )

        # Rename to expected output path neu khac
        downloaded = Path(downloaded_path)
        if downloaded != output_path:
            if output_path.exists():
                output_path.unlink()
            downloaded.rename(output_path)

        logger.success(f"Downloaded {model_type} to {output_path}")
        return True

    except ImportError:
        logger.warning("huggingface_hub not installed, fallback to URL download")
        return False
    except Exception as e:
        logger.error(f"HF Hub download failed for {model_type}: {e}")
        return False


def _download_from_url(
    model_type: str,
    output_path: Path,
    timeout: int = 120,
) -> bool:
    """Download model weights tu URL fallback.

    Args:
        model_type: Loai model (lstm, xgb-a, xgb-b).
        output_path: Path luu file output.
        timeout: Timeout seconds cho request.

    Returns:
        True neu download thanh cong, False neu that bai.
    """
    url = MODEL_URLS.get(model_type)
    if not url:
        logger.error(f"No URL for model type: {model_type}")
        return False

    logger.info(f"Downloading {model_type} from URL: {url}")

    try:
        import urllib.request

        # Download with progress
        def _report_progress(
            block_count: int,
            block_size: int,
            total_size: int,
        ) -> None:
            if total_size > 0:
                downloaded = block_count * block_size
                percent = min(100.0, downloaded * 100.0 / total_size)
                if block_count % 20 == 0:  # Print every ~20 blocks
                    mb_dl = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r  Progress: {percent:.1f}% "
                        f"({mb_dl:.2f}/{mb_total:.2f} MB)",
                        end="",
                        flush=True,
                    )

        logger.info(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path, reporthook=_report_progress)
        print()  # Newline after progress
        logger.success(f"Downloaded {model_type} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"URL download failed for {model_type}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_model(
    model_type: str,
    output_dir: Path,
    use_hf_hub: bool = True,
    verify_checksum: bool = True,
) -> bool:
    """Download mot model weight.

    Args:
        model_type: Loai model (lstm, xgb-a, xgb-b).
        output_dir: Directory output.
        use_hf_hub: Neu True, uu tien HF Hub truoc URL fallback.
        verify_checksum: Neu True, verify SHA256 checksum.

    Returns:
        True neu download thanh cong, False neu that bai.
    """
    model_name = MODEL_NAMES.get(model_type, model_type)
    filename = MODEL_URLS[model_type].split("/")[-1]
    output_path = output_dir / filename

    logger.info(f"Downloading {model_name} ({model_type}) -> {output_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    success = False
    if use_hf_hub:
        try:
            success = _download_with_hf_hub(model_type, output_path)
        except Exception:
            success = False

    if not success:
        success = _download_from_url(model_type, output_path)

    if not success:
        return False

    # Verify checksum
    if verify_checksum and MODEL_CHECKSUMS.get(model_type):
        expected = MODEL_CHECKSUMS[model_type]
        if expected:
            actual = _compute_sha256(output_path)
            if actual != expected:
                logger.error(
                    f"Checksum mismatch for {model_type}!\n"
                    f"  Expected: {expected}\n"
                    f"  Actual:   {actual}"
                )
                output_path.unlink()
                return False
            logger.info(f"Checksum verified for {model_type}")

    return True


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Command line arguments (default: sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download model weights cho Agentic Quant system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model-type all\n"
            "  %(prog)s --model-type lstm --output-dir ./models\n"
            "  %(prog)s --model-type xgb-a --no-verify\n"
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "xgb-a", "xgb-b", "all"],
        required=True,
        help="Loai model can download (lstm, xgb-a, xgb-b, all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/weights",
        help="Directory luu model weights (default: ./models/weights)",
    )

    parser.add_argument(
        "--no-hf-hub",
        action="store_true",
        help="Khong dung HuggingFace Hub, chi dung URL fallback",
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Khong verify checksum sau khi download",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download neu file da ton tai",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In log chi tiet (debug level)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args(argv)

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<level>{level: <8}</level> | {message}")

    # Resolve output directory
    output_dir = Path(args.output_dir).resolve()
    use_hf_hub = not args.no_hf_hub
    verify_checksum = not args.no_verify

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"HuggingFace Hub: {'enabled' if use_hf_hub else 'disabled'}")
    logger.info(f"Checksum verify: {'enabled' if verify_checksum else 'disabled'}")

    # Determine models to download
    if args.model_type == "all":
        models_to_download = ["lstm", "xgb-a", "xgb-b"]
    else:
        models_to_download = [args.model_type]

    logger.info(f"Models to download: {', '.join(models_to_download)}")

    # Download each model
    success_count = 0
    fail_count = 0

    for model_type in models_to_download:
        output_path = output_dir / MODEL_URLS[model_type].split("/")[-1]

        # Check if already exists
        if output_path.exists() and not args.force:
            logger.info(
                f"{model_type} da ton tai tai {output_path}, "
                f"skip (dung --force de re-download)"
            )
            success_count += 1
            continue

        start_time = time.perf_counter()
        ok = download_model(
            model_type=model_type,
            output_dir=output_dir,
            use_hf_hub=use_hf_hub,
            verify_checksum=verify_checksum,
        )
        elapsed = time.perf_counter() - start_time

        if ok:
            file_size = output_path.stat().st_size if output_path.exists() else 0
            logger.success(
                f"Downloaded {model_type} trong {elapsed:.2f}s "
                f"({file_size / (1024*1024):.2f} MB)"
            )
            success_count += 1
        else:
            logger.error(f"Failed to download {model_type} sau {elapsed:.2f}s")
            fail_count += 1

    # Summary
    total = len(models_to_download)
    logger.info(
        f"\n{'='*50}\n"
        f"Download Summary: {success_count}/{total} success, "
        f"{fail_count} failed\n"
        f"Output: {output_dir}\n"
        f"{'='*50}"
    )

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
