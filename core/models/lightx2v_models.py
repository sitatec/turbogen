import types
import random
import functools
from pathlib import Path
from typing import Literal, override

import torch
import numpy as np

from core.utils import is_hopper_gpu
from core.models.base_model import BaseModel, GenerationType
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v import LightX2VPipeline as LightX2VPipelineBase
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner


class _LightX2VPipeline(LightX2VPipelineBase):
    def enable_compilation(self, supported_shapes: list[list[int]]):
        pass  # TODO: implement

    @override
    @torch.no_grad()
    def generate(  # pyrefly: ignore
        self,
        seed,
        prompt,
        negative_prompt,
        save_result_path,
        image_path=None,
        last_frame_path=None,
        audio_path=None,
        src_ref_images=None,
        src_video=None,
        src_mask=None,
        return_result_tensor=False,
        height: int | None = None,
        width: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        duration_seconds: float | None = None,
    ) -> torch.Tensor | None:
        if seed is None or seed == -1:
            seed = random.randint(1, np.iinfo(np.int32).max)

        input_info = init_empty_input_info(self.task)

        update_input_info_from_dict(
            input_info,
            _AttrDict(
                {
                    **super().__dict__,
                    **{
                        "seed": seed,
                        "prompt": prompt,
                        "target_shape": [height, width],
                        "negative_prompt": negative_prompt,
                        "save_result_path": save_result_path,
                        "image_path": image_path,
                        "last_frame_path": last_frame_path,
                        "audio_path": audio_path,
                        "src_ref_images": src_ref_images,
                        "src_video": src_video,
                        "src_mask": src_mask,
                        "return_result_tensor": return_result_tensor,
                    },
                }
            ),
        )

        if guidance_scale or steps or duration_seconds:
            print(
                "⚠️WARNING⚠️: guidance_scale, duration and steps params should only be provided during testing or local runs. It will break think in concurrent envs like production inference."
            )
            self.runner.set_config(
                {
                    "target_video_length": (
                        int(self.runner.config.fps * duration_seconds) + 1
                        if duration_seconds
                        else self.runner.config.target_video_length
                    ),
                    "infer_steps": steps or self.infer_steps,
                    "sample_guide_scale": guidance_scale or self.sample_guide_scale,
                    "enable_cfg": guidance_scale > 1
                    if guidance_scale
                    else self.enable_cfg,
                }
            )

        result = self.runner.run_pipeline(input_info)
        is_video = self.task.endswith("2v") or self.task.endswith("2av")
        if is_video:
            return result["video"]
        return result[0].permute(0, 2, 3, 1).squeeze()


class _BaseLightx2vModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        model_name: str,
        model_cls: str,
        model_path: str,
        generation_type: GenerationType,
        aspect_ratios: dict[str, dict[str, tuple[int, int]]],
        attention_backend: Literal[
            "flash_attn3", "sage_attn2", "torch_sdpa"
        ] = "flash_attn3" if is_hopper_gpu() else "sage_attn2",
        infer_steps: int = 4,
        guidance_scale: float = 1,
        compile: bool = False,
        default_negative_prompt: str | None = None,
        lora_configs: list[dict] | None = None,
        supports_last_frame: bool = False,  # Only relevant for I2V gen type
        enable_cpu_offload: bool = False,
        quant_scheme: str | None = None,
        text_encoder_quantized: bool = False,
        quantized_model_path: str | None = None,
        quantized_text_encoder_path: str | None = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.supported_aspect_ratios = aspect_ratios
        self.default_negative_prompt = default_negative_prompt
        self.generation_type = generation_type
        self.default_inference_steps = infer_steps
        self.default_guidance_scale = guidance_scale

        self.pipe = _LightX2VPipeline(
            model_path=model_path,
            model_cls=model_cls,
            task=generation_type.value,
        )

        if enable_cpu_offload:
            self.pipe.enable_offload(
                text_encoder_offload=True,
                image_encoder_offload=True,
                vae_offload=True,
            )

        if (
            quant_scheme
            or quantized_model_path
            or quantized_text_encoder_path
            or text_encoder_quantized
        ):
            self.pipe.enable_quantize(
                quant_scheme=quant_scheme or "fp8-sgl",
                dit_quantized=(
                    quant_scheme is not None or quantized_model_path is not None
                ),
                dit_quantized_ckpt=quantized_model_path,
                text_encoder_quantized=text_encoder_quantized,
                text_encoder_quantized_ckpt=quantized_text_encoder_path,
            )

        if lora_configs:
            self.pipe.enable_lora(lora_configs)

        if model_cls == "qwen_image":
            self.pipe.text_encoder_type = "lightllm_kernel"  # pyrefly: ignore
            self.pipe.lightllm_config = {  # pyrefly: ignore
                "use_flash_attention_kernel": False,
                "use_rmsnorm_kernel": True,
            }

        self.pipe.create_generator(
            attn_mode=attention_backend,
            resize_mode="adaptive" if generation_type == "i2i" else None,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
        )

        if compile:
            self.pipe.enable_compilation(
                [
                    list(shape)
                    for resolutions in aspect_ratios.values()
                    for shape in resolutions.values()
                ]
            )

        if generation_type == GenerationType.I2V and supports_last_frame:
            # Add support for First-Last-Frame To Video
            def encode_input(self: DefaultRunner):
                if self.input_info and self.input_info.last_frame_path:
                    return self._run_input_encoder_local_flf2v()

                return self._run_input_encoder_local_i2v()

            self.pipe.runner.run_input_encoder = types.MethodType(
                encode_input, self.pipe.runner
            )

    @override
    def generate(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        image_paths: list[str] = [],
        last_frame_path: str | None = None,
        seed: int = -1,  # -1 for random seed
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        duration_seconds: float | None = None,
    ) -> torch.Tensor:
        image_paths_str = ",".join(image_paths)

        if aspect_ratio not in self.supported_aspect_ratios:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        if resolution not in self.supported_aspect_ratios[aspect_ratio]:
            raise ValueError(f"Invalid resolution: {resolution}")

        width, height = self.supported_aspect_ratios[aspect_ratio][resolution]

        return self.pipe.generate(  # pyrefly: ignore
            seed=seed,
            image_path=image_paths_str,
            prompt=prompt,
            negative_prompt=negative_prompt or self.default_negative_prompt,
            save_result_path=None,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            last_frame_path=last_frame_path,
            duration_seconds=duration_seconds,
            return_result_tensor=True,
        )


class QwenImageEditLite(_BaseLightx2vModel):
    def __init__(
        self,
        model_path: Path,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] = [],
        compile: bool = False,
        enable_cpu_offload: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_id="k",
            model_name="Qwen Image Edit 2511",
            model_cls="qwen-image-edit-2511",
            generation_type=GenerationType.I2I,
            model_path=str(model_path),
            compile=compile,
            quantized_model_path=quantized_model_path,
            lora_configs=[
                {
                    "path": f"{model_path}/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                    "strength": 1,
                },
                *lora_configs,
            ],
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=kwargs.pop("infer_steps", 4),
            aspect_ratios={
                "1:1": {"1K": (1024, 1024)},
                "16:9": {"1K": (1344, 768)},
                "9:16": {"1K": (768, 1344)},
                "4:3": {"1K": (1152, 896)},
                "3:4": {"1K": (896, 1152)},
                "3:2": {"1K": (1216, 832)},
                "2:3": {"1K": (832, 1216)},
                "5:4": {"1K": (1088, 896)},
                "4:5": {"1K": (896, 1088)},
                "21:9": {"1K": (1536, 640)},
                "9:21": {"1K": (640, 1536)},
            },
            **kwargs,
        )

        self.max_input_images = 3


class QwenImageLite(_BaseLightx2vModel):
    def __init__(
        self,
        model_path: Path,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] = [],
        text_encoder=None,
        vae=None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        **kwargs,
    ):
        if text_encoder and vae:
            QwenImageRunner.load_model = self._create_patched_load_model(
                text_encoder, vae
            )
        super().__init__(
            model_id="l",
            model_name="Qwen Image 2512",
            model_cls="qwen-image-2512",
            generation_type=GenerationType.T2I,
            model_path=str(model_path),
            compile=compile,
            quantized_model_path=quantized_model_path,
            lora_configs=[
                {
                    "path": f"{model_path}/lora/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
                    "strength": 1,
                },
                *lora_configs,
            ],
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=kwargs.pop("infer_steps", 4),
            aspect_ratios={
                "1:1": {"1.3K": (1328, 1328)},
                "16:9": {"1.3K": (1664, 928)},
                "9:16": {"1.3K": (928, 1664)},
                "4:3": {"1.3K": (1472, 1104)},
                "3:4": {"1.3K": (1104, 1472)},
                "3:2": {"1.3K": (1584, 1056)},
                "2:3": {"1.3K": (1056, 1584)},
                "5:4": {"1.3K": (1440, 1152)},
                "4:5": {"1.3K": (1152, 1440)},
            },
            **kwargs,
        )

    def _create_patched_load_model(self, text_encoder, vae):
        original_load_model = QwenImageRunner.load_model

        @functools.wraps(original_load_model)
        def patched_load_model(self):
            orig_load_text_encoder = self.load_text_encoder
            orig_load_vae = self.load_vae

            @functools.wraps(orig_load_text_encoder)
            def _load_text_encoder(_):
                return text_encoder

            @functools.wraps(orig_load_vae)
            def _load_vae(_):
                return vae

            self.load_text_encoder = types.MethodType(_load_text_encoder, self)
            self.load_vae = types.MethodType(_load_vae, self)

            try:
                return original_load_model(self)
            finally:
                # Restore original behavior at class level
                QwenImageRunner.load_model = original_load_model

        return patched_load_model


class ZImageTurbo(_BaseLightx2vModel):
    def __init__(
        self,
        model_path: Path,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] | None = None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_id="0",
            model_name="Z-Image-Turbo",
            model_cls="z_image",
            generation_type=GenerationType.T2I,
            model_path=str(model_path),
            compile=compile,
            quantized_model_path=quantized_model_path,
            lora_configs=lora_configs,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=kwargs.pop("infer_steps", 9),
            guidance_scale=kwargs.pop("guidance_scale", 0),
            aspect_ratios={
                "1:1": {"1K": (1024, 1024), "1.3K": (1280, 1280), "1.5K": (1536, 1536)},
                "16:9": {"1K": (1344, 768), "1.3K": (1536, 864), "1.5K": (2048, 1152)},
                "9:16": {"1K": (768, 1344), "1.3K": (864, 1536), "1.5K": (1152, 2048)},
                "4:3": {"1K": (1152, 896), "1.3K": (1472, 1104), "1.5K": (1728, 1296)},
                "3:4": {"1K": (896, 1152), "1.3K": (1104, 1472), "1.5K": (1296, 1728)},
                "3:2": {"1K": (1216, 832), "1.3K": (1536, 1024), "1.5K": (1872, 1248)},
                "2:3": {"1K": (832, 1216), "1.3K": (1024, 1536), "1.5K": (1248, 1872)},
                "5:4": {"1K": (1088, 896), "1.3K": (1360, 1088), "1.5K": (1680, 1344)},
                "4:5": {"1K": (896, 1088), "1.3K": (1088, 1360), "1.5K": (1344, 1680)},
                "21:9": {"1K": (1536, 640), "1.3K": (1680, 720), "1.5K": (2016, 864)},
                "9:21": {"1K": (640, 1536), "1.3K": (720, 1680), "1.5K": (864, 2016)},
            },
            **kwargs,
        )


class Wan22Lite(_BaseLightx2vModel):
    def __init__(
        self,
        model_path: Path,
        lora_configs: list[dict] | None = None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        quant_scheme: str | None = "fp8-sgl",
        text_encoder_quantized: bool = True,
        generation_type: Literal[
            GenerationType.I2V, GenerationType.T2V
        ] = GenerationType.I2V,
        **kwargs,
    ):
        super().__init__(
            model_id="m",
            model_name=f"Wan 2.2 A14B {generation_type.value.upper()}",
            model_cls="wan2.2_moe_distill",
            generation_type=generation_type,
            supports_last_frame=True,
            model_path=str(model_path),
            compile=compile,
            attention_backend="sage_attn2",
            quant_scheme=quant_scheme,
            text_encoder_quantized=text_encoder_quantized,
            lora_configs=lora_configs,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=kwargs.pop("infer_steps", 4),
            guidance_scale=kwargs.pop("guidance_scale", 1),
            aspect_ratios={
                "16:9": {"480p": (854, 480), "720p": (1280, 720)},
                "9:16": {"480p": (480, 854), "720p": (720, 1280)},
            },
            **kwargs,
        )


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


__all__ = ["Wan22Lite", "ZImageTurbo", "QwenImageLite", "QwenImageEditLite"]
