from pathlib import Path

import huggingface_hub


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
    qwen_image_edit_2511_path = project_root / "models/Qwen-Image-Edit-2511-Lightning"
    qwen_image_2512_path = project_root / "models/Qwen-Image-2512-Lightning"

    # Qwen-Image-Edit-2511-Lightning
    huggingface_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        local_dir=qwen_image_edit_2511_path,
    )
    huggingface_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        local_dir=qwen_image_edit_2511_path,
        allow_patterns=[
            "text_encoder/**",
            "vae/**",
            "scheduler/**",
            "tokenizer/**",
            "processor/**",
        ],
    )
    # Qwen-Image-2512-Lightning
    huggingface_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-2512-Lightning",
        filename="qwen_image_2512_fp8_e4m3fn_scaled_4steps_v1.0.safetensors",
        local_dir=qwen_image_2512_path,
    )

    return qwen_image_edit_2511_path.resolve(), qwen_image_2512_path.resolve()
