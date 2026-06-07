from typing import Literal
import os
from pathlib import Path

import huggingface_hub as hf_hub

_ROOT_DIR = (
    Path(os.environ["MODEL_WEIGHTS_ROOT_DIR"])
    if os.environ.get("MODEL_WEIGHTS_ROOT_DIR")
    else Path(__file__).parent / "_model_weights"
)

_SCORING_MODELS_DIR = (_ROOT_DIR / "scoring_models").resolve()


def download_qwen_image(
    te_quant_method: Literal["gptq", "bnb"] | None = "gptq",
    offline: bool = False,
) -> Path:
    qwen_image_2512_path = _ROOT_DIR / "Qwen-Image-2512-Fast"

    if offline and qwen_image_2512_path.exists():
        return qwen_image_2512_path

    ignore_patterns = ["transformer/**", "scheduler/**"]
    if te_quant_method:
        ignore_patterns.append("text_encoder/**")

    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-2512",
        local_dir=qwen_image_2512_path,
        ignore_patterns=ignore_patterns,
    )
    hf_hub.snapshot_download(
        repo_id="sitatech/Qwen-Image-2512-Turbo-2stpes-FP8",
        local_dir=qwen_image_2512_path,  # download transformer and scheduler sub-folders
    )
    if te_quant_method:
        hf_hub.snapshot_download(
            repo_id=f"sitatech/Qwen2.5-VL-7B-Instruct-{te_quant_method.upper()}-Int4",
            local_dir=qwen_image_2512_path / "text_encoder",
        )

    return qwen_image_2512_path.resolve()


def download_qwen_image_edit(
    te_quant_method: Literal["gptq", "bnb"] | None = "gptq",
    offline: bool = False,
) -> Path:
    qwen_image_edit_2511_path = _ROOT_DIR / "Qwen-Image-Edit-2511-Lightning"

    if offline and qwen_image_edit_2511_path.exists():
        return qwen_image_edit_2511_path

    ignore_patterns = ["transformer/diffusion_pytorch_model*"]
    if te_quant_method:
        ignore_patterns.append("text_encoder/**")

    hf_hub.snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        local_dir=qwen_image_edit_2511_path,
        ignore_patterns=ignore_patterns,
    )

    hf_hub.hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename="qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        local_dir=qwen_image_edit_2511_path / "transformer",
    )

    if te_quant_method:
        hf_hub.snapshot_download(
            repo_id=f"sitatech/Qwen2.5-VL-7B-Instruct-{te_quant_method.upper()}-Int4",
            local_dir=qwen_image_edit_2511_path / "text_encoder",
        )

    return qwen_image_edit_2511_path.resolve()


def download_zimage_models(
    te_quant_method: Literal["gptq", "bnb"] | None = "gptq",
    dit_quant_method: Literal["fp8"] | None = None,
    offline: bool = False,
):
    zimage_turbo_path = _ROOT_DIR / "Z-Image-Turbo"

    if offline and zimage_turbo_path.exists():
        return zimage_turbo_path

    ignore_patterns = ["assets/**"]
    if dit_quant_method:
        ignore_patterns.append("transformer/diffusion_pytorch_model*")
    if te_quant_method:
        ignore_patterns.append("text_encoder/**")

    hf_hub.snapshot_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        local_dir=zimage_turbo_path,
        ignore_patterns=ignore_patterns,
    )

    if dit_quant_method:
        hf_hub.hf_hub_download(
            repo_id="lightx2v/Z-Image-Turbo-Quantized",
            filename="z_image_turbo_scaled_fp8_e4m3fn.safetensors",
            local_dir=zimage_turbo_path / "transformer",
        )

    if te_quant_method:
        hf_hub.snapshot_download(
            repo_id="JunHowie/Qwen3-4B-GPTQ-Int4" if te_quant_method == "gptq" else "unsloth/Qwen3-4B-bnb-4bit",
            local_dir=zimage_turbo_path / "text_encoder",
        )

    return zimage_turbo_path.resolve()


def download_wan22_i2v_model(offline: bool = False, dit_quant_method: Literal["fp8", "nvfp4"] | None = "fp8"):
    wan22_i2v_path = _ROOT_DIR / "Wan2.2-I2V"

    if offline and wan22_i2v_path.exists():
        return wan22_i2v_path

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
    if dit_quant_method == "nvfp4":
        hf_hub.hf_hub_download(
            repo_id="lightx2v/Wan2.2-NVFP4-Sparse",
            filename="Wan2.2-I2V-A14B_NVFP4_Sparse_high.safetensors",
            local_dir=wan22_i2v_path / "high_noise_model",
        )
        hf_hub.hf_hub_download(
            repo_id="lightx2v/Wan2.2-NVFP4-Sparse",
            filename="Wan2.2-I2V-A14B_NVFP4_Sparse_low.safetensors",
            local_dir=wan22_i2v_path / "low_noise_model",
        )
    else:
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

    return wan22_i2v_path.resolve()


def download_wan22_t2v_models(offline: bool = False, dit_quant_method: Literal["fp8", "nvfp4"] | None = "fp8"):
    wan22_t2v_path = _ROOT_DIR / "Wan2.2-T2V"

    if offline and wan22_t2v_path.exists():
        return wan22_t2v_path

    hf_hub.snapshot_download(
        repo_id="lightx2v/Encoders",
        local_dir=wan22_t2v_path,
        allow_patterns=["google/**", "models_t5_umt5-xxl-enc-fp8.pth"],
    )
    hf_hub.hf_hub_download(
        repo_id="lightx2v/Autoencoders",
        filename="Wan2.1_VAE.pth",
        local_dir=wan22_t2v_path,
    )
    hf_hub.hf_hub_download(
        repo_id="sitatech/Wan2.2-FP8-Models",
        filename="config.json",
        local_dir=wan22_t2v_path,
    )

    if dit_quant_method == "nvfp4":
        hf_hub.hf_hub_download(
            repo_id="lightx2v/Wan2.2-NVFP4-Sparse",
            filename="Wan2.2-T2V-A14B_NVFP4_Sparse_high.safetensors",
            local_dir=wan22_t2v_path / "high_noise_model",
        )
        hf_hub.hf_hub_download(
            repo_id="lightx2v/Wan2.2-NVFP4-Sparse",
            filename="Wan2.2-T2V-A14B_NVFP4_Sparse_low.safetensors",
            local_dir=wan22_t2v_path / "low_noise_model",
        )
    else:
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

    return wan22_t2v_path.resolve()


def download_video_scorer(offline: bool = False) -> Path:
    """
    Download the VideoReward model.
    """
    video_reward_path = _SCORING_MODELS_DIR / "VideoReward"

    if offline and video_reward_path.exists():
        return video_reward_path

    hf_hub.snapshot_download(
        repo_id="sitatech/VideoReward",
        local_dir=video_reward_path,
    )

    return video_reward_path.resolve()


def download_image_scorer(
    quant_method: Literal["gptq", "bnb"] | None = "gptq",
    offline: bool = False,
) -> Path:
    """
    Download the TianheWu/VisualQuality-R1-7B, aesthetic-predictor-v2 and clip-vit-large-patch14 models.
    """
    image_scorer_path = _SCORING_MODELS_DIR / "image_scorer"

    if offline and image_scorer_path.exists():
        return image_scorer_path

    hf_hub.snapshot_download(
        repo_id=(
            f"sitatech/VisualQuality-R1-7B-{quant_method.upper()}-Int4"
            if quant_method
            else "TianheWu/VisualQuality-R1-7B"
        ),
        local_dir=image_scorer_path / "visual_quality_r1",
    )

    hf_hub.snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir=image_scorer_path / "clip-vit-l14",
    )

    hf_hub.hf_hub_download(
        repo_id="sitatech/aesthetic-predictor-v2",
        filename="sac+logos+ava1-l14-linearMSE.pth",
        local_dir=image_scorer_path,
    )

    return image_scorer_path.resolve()


def download_prompt_enhancer(
    quant_method: Literal["gptq", "bnb"] | None = "gptq",
    offline: bool = False,
) -> Path:
    model_path = _ROOT_DIR / "prompt_enhancer"

    if offline and model_path.exists():
        return model_path

    hf_hub.snapshot_download(
        repo_id=(
            f"sitatech/Qwen3-VL-8B-Instruct-{quant_method.upper()}-Int4"
            if quant_method
            else "Qwen/Qwen3-VL-8B-Instruct"
        ),
        local_dir=model_path,
    )

    return model_path.resolve()


def download_nsfw_model(offline: bool = False):
    model_path = _ROOT_DIR / "nsfw_model"

    if offline and model_path.exists():
        return model_path

    hf_hub.snapshot_download(repo_id="Freepik/nsfw_image_detector", local_dir=model_path)

    return model_path.resolve()


def _symlink_common_components(
    source_dir: Path,
    destination_dir: Path,
    subdirs: list[str],
):
    source_dir = source_dir.resolve()

    for subdir in subdirs:
        dest = destination_dir / subdir
        src = source_dir / subdir
        if not dest.exists():
            dest.symlink_to(src)
