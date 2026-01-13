from pathlib import Path

import huggingface_hub as hf_hub

_ROOT_DIR = Path(__file__).parent


def download_qwen_models() -> tuple[Path, Path]:
    """
    Download Qwen-Image-Edit-2511 and Qwen-Image-2512 models.

    Qwen-Image-Edit-2511 has the same text encoder and vae as Qwen-Image-2512.
    So we are not downloading them for Qwen-Image-2512, meaning that you will
    need to get them from the image edit model folder if you need them. But you
    could just let the image edit load them then share the same instances to save memory.

    Returns:
        tuple[Path, Path]: [image edit path, image path].
    """
    qwen_image_edit_2511_path = (
        _ROOT_DIR / "_model_weights/Qwen-Image-Edit-2511-Lightning"
    )
    qwen_image_2512_path = _ROOT_DIR / "_model_weights/Qwen-Image-2512-Lightning"

    # Qwen-Image-Edit-2511-Lightning
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
        local_dir=qwen_image_edit_2511_path / "lora",
    )
    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        local_dir=qwen_image_edit_2511_path,
    )
    # Qwen-Image-2512-Lightning
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-2512-Lightning",
        filename="Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
        local_dir=qwen_image_2512_path / "lora",
    )
    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-2512",
        local_dir=qwen_image_2512_path,
        allow_patterns=["transformer/**"],
    )
    _symlink_common_components(qwen_image_2512_path, qwen_image_edit_2511_path)

    return qwen_image_edit_2511_path.resolve(), qwen_image_2512_path.resolve()


def _symlink_common_components(
    destination_dir: Path,
    source_dir: Path,
):
    text_encoder_symlink = destination_dir / "text_encoder"
    vae_symlink = destination_dir / "vae"
    scheduler_symlink = destination_dir / "scheduler"
    tokenizer_symlink = destination_dir / "tokenizer"

    if not text_encoder_symlink.exists():
        text_encoder_symlink.symlink_to(
            source_dir / "text_encoder", target_is_directory=True
        )
    if not vae_symlink.exists():
        vae_symlink.symlink_to(source_dir / "vae", target_is_directory=True)
    if not scheduler_symlink.exists():
        scheduler_symlink.symlink_to(source_dir / "scheduler", target_is_directory=True)
    if not tokenizer_symlink.exists():
        tokenizer_symlink.symlink_to(source_dir / "tokenizer", target_is_directory=True)


def download_zimage_models():
    zimage_turbo_path = _ROOT_DIR / "_model_weights/Z-Image-Turbo"

    hf_hub.snapshot_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        local_dir=zimage_turbo_path,
        ignore_patterns=["assets"],
    )

    return zimage_turbo_path.resolve()


def download_wan22_models():
    wan22_5b_path = _ROOT_DIR / "_model_weights/Wan2.2"

    hf_hub.snapshot_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=wan22_5b_path,
        ignore_patterns=["assets", "examples"],
    )

    return wan22_5b_path.resolve()
