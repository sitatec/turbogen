from pathlib import Path

import huggingface_hub as hf_hub


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
    project_root = Path(__file__).parent
    qwen_image_edit_2511_path = (
        project_root / "_model_weights/Qwen-Image-Edit-2511-Lightning"
    )
    qwen_image_2512_path = project_root / "_model_weights/Qwen-Image-2512-Lightning"

    # Qwen-Image-Edit-2511-Lightning
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        local_dir=qwen_image_edit_2511_path / "lora",
    )
    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        local_dir=qwen_image_edit_2511_path,
    )

    # Qwen-Image-2512-Lightning
    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-2512",
        local_dir=qwen_image_2512_path,
        allow_patterns=["transformer/config.json"],
    )
    hf_hub.hf_hub_download(
        repo_id="sitatech/Qwen-Image-FP8-Models",
        filename="Qwen-Image-2512-FP8.safetensors",
        local_dir=qwen_image_2512_path / "fp8",
    )
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-2512-Lightning",
        filename="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        local_dir=qwen_image_2512_path / "lora",
    )

    symlink_common_components(qwen_image_2512_path, qwen_image_edit_2511_path)

    return qwen_image_edit_2511_path.resolve(), qwen_image_2512_path.resolve()


def symlink_common_components(
    qwen_image_2512_path: Path,
    qwen_image_edit_2511_path: Path,
):
    text_encoder_symlink = qwen_image_2512_path / "text_encoder"
    vae_symlink = qwen_image_2512_path / "vae"
    scheduler_symlink = qwen_image_2512_path / "scheduler"
    tokenizer_symlink = qwen_image_2512_path / "tokenizer"

    if not text_encoder_symlink.exists():
        text_encoder_symlink.symlink_to(
            qwen_image_edit_2511_path / "text_encoder", target_is_directory=True
        )
    if not vae_symlink.exists():
        vae_symlink.symlink_to(
            qwen_image_edit_2511_path / "vae", target_is_directory=True
        )
    if not scheduler_symlink.exists():
        scheduler_symlink.symlink_to(
            qwen_image_edit_2511_path / "scheduler", target_is_directory=True
        )
    if not tokenizer_symlink.exists():
        tokenizer_symlink.symlink_to(
            qwen_image_edit_2511_path / "tokenizer", target_is_directory=True
        )
