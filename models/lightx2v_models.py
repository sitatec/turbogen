from lightx2v.models.runners.default_runner import DefaultRunner
import types
import functools
from typing import Literal, override
import random

import torch
import numpy as np

from lightx2v.utils.input_info import I2IInputInfo
from lightx2v.utils.input_info import T2VInputInfo
from lightx2v.utils.input_info import T2IInputInfo
from lightx2v.utils.input_info import set_input_info
from lightx2v import LightX2VPipeline as LightX2VPipelineBase
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner


class LightX2VPipeline(LightX2VPipelineBase):
    def enable_compilation(self, supported_shapes: list[list[int]]):
        pass  # TODO: implement

    @override
    @torch.no_grad()
    def generate(
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
        guidance_scale: int | None = None,
    ):
        if seed is None or seed == -1:
            seed = random.randint(1, np.iinfo(np.int32).max)

        input_info = set_input_info(
            {
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "save_result_path": save_result_path,
                "image_path": image_path,
                "last_frame_path": last_frame_path,
                "audio_path": audio_path,
                "src_ref_images": src_ref_images,
                "src_video": src_video,
                "src_mask": src_mask,
                "return_result_tensor": return_result_tensor,
            }
        )

        if (
            isinstance(input_info, (T2IInputInfo, T2VInputInfo, I2IInputInfo))
            and height
            and width
        ):
            input_info.custom_shape = [height, width]

        if guidance_scale or steps:
            print(
                "⚠️WARNING⚠️: guidance_scale and steps params should only be provided during testing or local runs. It will break think in concurrent envs like production inference."
            )
            self.runner.set_config(
                {
                    "infer_steps": steps or self.infer_steps,
                    "sample_guide_scale": guidance_scale or self.sample_guide_scale,
                    "enable_cfg": guidance_scale > 1
                    if guidance_scale
                    else self.enable_cfg,
                }
            )

        return self.runner.run_pipeline(input_info)


class BaseModel:
    def __init__(
        self,
        model_cls: str,
        model_path: str,
        generation_type: Literal["t2i", "i2i", "i2v", "ti2v"],
        aspect_ratios: dict[str, dict[str, tuple[int, int]]],
        attention_backend: Literal["flash_attn3", "sage_attn2"] = "flash_attn3",
        quantized_model_path: str | None = None,
        infer_steps: int = 8,
        guidance_scale: int = 1,
        compile: bool = False,
        default_negative_prompt: str | None = None,
        lora_configs: list[dict] | None = None,
        quant_scheme: str | None = None,
        enable_cpu_offload: bool = False,
    ):
        self.pipe = LightX2VPipeline(
            model_path=model_path,
            model_cls=model_cls,
            task=generation_type,
        )

        if enable_cpu_offload:
            self.pipe.enable_offload(
                text_encoder_offload=True,
                image_encoder_offload=True,
                vae_offload=True,
            )

        if quant_scheme or quantized_model_path:
            self.pipe.enable_quantize(
                dit_quantized=True,
                dit_quantized_ckpt=quantized_model_path,
                quant_scheme=quant_scheme or "fp8-sgl",
            )

        if lora_configs:
            self.pipe.enable_lora(lora_configs)

        self.pipe.create_generator(
            attn_mode=attention_backend,
            resize_mode="adaptive" if generation_type == "i2i" else None,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
        )

        self.aspect_ratios = aspect_ratios
        self.default_negative_prompt = default_negative_prompt

        if compile:
            self.pipe.enable_compilation(
                [
                    list(shape)
                    for resolutions in aspect_ratios.values()
                    for shape in resolutions.values()
                ]
            )

        if generation_type == "ti2v" and model_cls == "wan2.2":

            def unified_ti2v_input_encoder(self: DefaultRunner):
                assert self.input_info, (
                    f"Oops, this shouldn't happen, input_info is not defined or is empty: {self.input_info}"
                )
                if self.input_info.image_path:
                    if self.input_info.last_frame_path:
                        return self._run_input_encoder_local_flf2v()

                    return self._run_input_encoder_local_i2v()

                return self._run_input_encoder_local_t2v()

            self.pipe.runner.run_input_encoder = types.MethodType(
                unified_ti2v_input_encoder, self.pipe.runner
            )

    def generate(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        output_path: str | None = None,
        image_paths: list[str] = [],
        last_frame_path: str | None = None,
        seed: int = -1,  # -1 for random seed
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: int | None = None,
    ):
        image_paths_str = ",".join(image_paths)

        if aspect_ratio not in self.aspect_ratios:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        if resolution not in self.aspect_ratios[aspect_ratio]:
            raise ValueError(f"Invalid resolution: {resolution}")

        width, height = self.aspect_ratios[aspect_ratio][resolution]

        self.pipe.generate(
            seed=seed,
            image_path=image_paths_str,
            prompt=prompt,
            negative_prompt=negative_prompt or self.default_negative_prompt,
            save_result_path=output_path,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            last_frame_path=last_frame_path,
        )


class QwenImageEdit(BaseModel):
    def __init__(
        self,
        model_path: str,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] | None = None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_cls="qwen-image-edit-2511",
            generation_type="i2i",
            model_path=model_path,
            compile=compile,
            quantized_model_path=quantized_model_path,
            lora_configs=lora_configs,
            enable_cpu_offload=enable_cpu_offload,
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


class QwenImage(BaseModel):
    def __init__(
        self,
        model_path: str,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] | None = None,
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
            model_cls="qwen-image-2512",
            generation_type="t2i",
            model_path=model_path,
            compile=compile,
            quantized_model_path=quantized_model_path,
            lora_configs=lora_configs,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=4,
            aspect_ratios={
                "1:1": {"1K": (1024, 1024), "1.3K": (1328, 1328)},
                "16:9": {"1K": (1344, 768), "1.3K": (1664, 928)},
                "9:16": {"1K": (768, 1344), "1.3K": (928, 1664)},
                "4:3": {"1K": (1152, 896), "1.3K": (1472, 1104)},
                "3:4": {"1K": (896, 1152), "1.3K": (1104, 1472)},
                "3:2": {"1K": (1216, 832), "1.3K": (1584, 1056)},
                "2:3": {"1K": (832, 1216), "1.3K": (1056, 1584)},
                "5:4": {"1K": (1088, 896), "1.3K": (1440, 1152)},
                "4:5": {"1K": (896, 1088), "1.3K": (1152, 1440)},
                "21:9": {"1K": (1536, 640), "1.3K": (1792, 768)},
                "9:21": {"1K": (640, 1536), "1.3K": (768, 1792)},
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


class ZImageTurbo(BaseModel):
    def __init__(
        self,
        model_path: str,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] | None = None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_cls="z_image",
            generation_type="t2i",
            model_path=model_path,
            compile=compile,
            quantized_model_path=quantized_model_path,
            lora_configs=lora_configs,
            enable_cpu_offload=enable_cpu_offload,
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


class Wan22_5B(BaseModel):
    def __init__(
        self,
        model_path: str,
        quantized_model_path: str | None = None,
        lora_configs: list[dict] | None = None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_cls="wan2.2",
            generation_type="ti2v",
            model_path=model_path,
            compile=compile,
            attention_backend="sage_attn2",
            quantized_model_path=quantized_model_path,
            lora_configs=lora_configs,
            enable_cpu_offload=enable_cpu_offload,
            aspect_ratios={
                "16:9": {"480p": (854, 480), "720p": (1280, 720)},
                "9:16": {"480p": (480, 854), "720p": (720, 1280)},
            },
            **kwargs,
        )


__all__ = [Wan22_5B, ZImageTurbo, QwenImage, QwenImageEdit]
