import os
import sys
from pathlib import Path

app_root_dir = Path(__file__).parent.resolve()

# Adding turbogen to system path for local script runs
sys.path.insert(
    0,
    str(app_root_dir / "turbogen"),
)

os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
weights_dir = app_root_dir / "weights"
os.environ["MODEL_WEIGHTS_ROOT_DIR"] = os.environ.get("MODEL_WEIGHTS_ROOT_DIR", str(weights_dir))

os.environ["HF_HOME"] = str(weights_dir / "huggingface")
os.environ["HF_HUB_CACHE"] = str(weights_dir / "huggingface" / "hub")
os.environ["KERNELS_CACHE"] = str(weights_dir / "community-kernels")


from turbogen.model_downloads import (  # noqa
    download_zimage_models as _download_zimage_models,
    download_qwen_image as _download_qwen_image,
    download_qwen_image_edit as _download_qwen_image_edit,
    download_wan22_i2v_model as _download_wan22_i2v_model,
    download_wan22_t2v_models as _download_wan22_t2v_models,
)


def download_zimage_models():
    return _download_zimage_models(offline=True, dit_quant_method="fp8")


def download_qwen_image():
    return _download_qwen_image(offline=True)


def download_qwen_image_edit():
    return _download_qwen_image_edit(offline=True)


def download_wan22_i2v_model():
    return _download_wan22_i2v_model(offline=True)


def download_wan22_t2v_models():
    return _download_wan22_t2v_models(offline=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downloads.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    # When running as a script to download, we want to allow internet access to Hugging Face
    os.environ["HF_HUB_OFFLINE"] = "0"

    print(f"Downloading model for: {model_name}...")
    if model_name == "wan22_i2v":
        download_wan22_i2v_model()
    elif model_name == "wan22_t2v":
        download_wan22_t2v_models()
    elif model_name == "z_image_turbo":
        download_zimage_models()
    elif model_name == "qwen_image":
        download_qwen_image()
    elif model_name == "qwen_image_edit":
        download_qwen_image_edit()
    else:
        print(f"Error: Unknown model {model_name}")
        sys.exit(1)
    print("Download completed successfully.")
