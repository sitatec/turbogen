import os
from pathlib import Path

import huggingface_hub as hf_hub

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

_ROOT_DIR = Path(__file__).parent / "_model_weights"

_SCORING_MODELS_DIR = (_ROOT_DIR / "scoring_models").resolve()


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
    qwen_image_edit_2511_path = _ROOT_DIR / "Qwen-Image-Edit-2511-Lightning"
    qwen_image_2512_path = _ROOT_DIR / "Qwen-Image-2512-Lightning"

    # Qwen-Image-Edit-2511-Lightning
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
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
    _symlink_common_components(
        qwen_image_edit_2511_path,
        qwen_image_2512_path,
        ["tokenizer", "scheduler", "vae", "text_encoder"],
    )

    return qwen_image_edit_2511_path.resolve(), qwen_image_2512_path.resolve()


def download_zimage_models():
    zimage_turbo_path = _ROOT_DIR / "Z-Image-Turbo"

    hf_hub.snapshot_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        local_dir=zimage_turbo_path,
        ignore_patterns=["assets"],
    )

    return zimage_turbo_path.resolve()


def download_wan22_models():
    wan22_i2v_path = _ROOT_DIR / "Wan2.2-I2V"
    wan22_t2v_path = _ROOT_DIR / "Wan2.2-T2V"

    hf_hub.snapshot_download(
        repo_id="lightx2v/Encoders",
        local_dir=wan22_i2v_path,
        allow_patterns=["google/**", "models_t5_umt5-xxl-enc-fp8.pth"],
    )
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Autoencoders",
        filename="Wan2.1_VAE.pth",
        local_dir=wan22_i2v_path,
    )
    hf_hub.hf_hub_download(
        repo_id="sitatech/Wan2.2-FP8-Models",
        filename="config.json",
        local_dir=wan22_i2v_path,
    )

    hf_hub.hf_hub_download(
        repo_id="sitatech/Wan2.2-FP8-Models",
        filename="wan2.2_i2v_A14b_high_noise_scaled_fp8_lightx2v_4step_1022.safetensors",
        local_dir=wan22_i2v_path / "high_noise_model",
    )
    hf_hub.hf_hub_download(
        repo_id="sitatech/Wan2.2-FP8-Models",
        filename="wan2.2_i2v_A14b_low_noise_scaled_fp8_lightx2v_4step_1022.safetensors",
        local_dir=wan22_i2v_path / "low_noise_model",
    )
    hf_hub.hf_hub_download(
        repo_id="sitatech/Wan2.2-FP8-Models",
        filename="wan2.2_t2v_A14b_high_noise_scaled_fp8_lightx2v_4step_1217.safetensors",
        local_dir=wan22_t2v_path / "high_noise_model",
    )
    hf_hub.hf_hub_download(
        repo_id="sitatech/Wan2.2-FP8-Models",
        filename="wan2.2_t2v_A14b_low_noise_scaled_fp8_lightx2v_4step_1217.safetensors",
        local_dir=wan22_t2v_path / "low_noise_model",
    )

    _symlink_common_components(
        wan22_i2v_path,
        wan22_t2v_path,
        [
            "google",
            "models_t5_umt5-xxl-enc-fp8.pth",
            "Wan2.1_VAE.pth",
            "config.json",
        ],
    )

    return wan22_i2v_path.resolve(), wan22_t2v_path.resolve()


def download_video_scorer() -> Path:
    """
    Download the VideoReward model.
    """
    video_reward_path = _SCORING_MODELS_DIR / "VideoReward"

    hf_hub.snapshot_download(
        repo_id="sitatech/VideoReward",
        local_dir=video_reward_path,
    )

    return video_reward_path.resolve()


def download_image_scorer() -> Path:
    """
    Download the TianheWu/VisualQuality-R1-7B, aesthetic-predictor-v2 and clip-vit-large-patch14-336 models.
    """
    image_scorer_path = _SCORING_MODELS_DIR / "image_scorer"

    hf_hub.snapshot_download(
        repo_id="TianheWu/VisualQuality-R1-7B",
        local_dir=image_scorer_path / "visual_quality_r1",
    )

    hf_hub.snapshot_download(
        repo_id="openai/clip-vit-large-patch14-336",
        local_dir=image_scorer_path / "clip-vit-l14",
    )

    hf_hub.hf_hub_download(
        repo_id="sitatech/aesthetic-predictor-v2",
        filename="sac+logos+ava1-l14-linearMSE.pth",
        local_dir=image_scorer_path,
    )

    return image_scorer_path.resolve()


def download_prompt_enhancer() -> Path:
    """
    Download the Qwen3-VL-8B-Instruct model.
    """
    model_path = _ROOT_DIR / "prompt_enhancer"

    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir=model_path,
    )

    return model_path.resolve()


def _symlink_common_components(
    source_dir: Path,
    destination_dir: Path,
    subdirs: list[str],
):
    for subdir in subdirs:
        dest = destination_dir / subdir
        src = source_dir / subdir
        if not dest.exists():
            dest.symlink_to(src)
