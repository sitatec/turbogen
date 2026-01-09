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
from lightx2v.utils.utils import seed_all
from lightx2v import LightX2VPipeline as LightX2VPipelineBase
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner


class LightX2VPipeline(LightX2VPipelineBase):
    def enable_compilation(self, supported_shapes: list[list[int]]):
        self.compile = True
        self.compile_shapes = supported_shapes

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
        height=None,
        width=None,
    ):
        if seed is None or seed == -1:
            seed = random.randint(1, np.iinfo(np.int32).max)

        # Run inference (following LightX2V pattern)
        self.seed = seed
        self.image_path = image_path
        self.last_frame_path = last_frame_path
        self.audio_path = audio_path
        self.src_ref_images = src_ref_images
        self.src_video = src_video
        self.src_mask = src_mask
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.save_result_path = save_result_path
        self.return_result_tensor = return_result_tensor
        seed_all(self.seed)

        input_info = set_input_info(self)
        if (
            isinstance(input_info, (T2IInputInfo, T2VInputInfo, I2IInputInfo))
            and height
            and width
        ):
            input_info.custom_shape = [height, width]
        else:
            self.target_height, self.target_width = height, width
            self.runner.set_config({"target_height": height, "target_width": width})

        return self.runner.run_pipeline(input_info)


class BaseModel:
    def __init__(
        self,
        model_cls: str,
        model_path: str,
        generation_type: Literal["t2i", "i2i", "i2v"],
        aspect_ratios: dict[str, dict[str, tuple[int, int]]],
        attention_backend: Literal["flash_attn3", "sage_attn2"] = "flash_attn3",
        infer_steps: int = 8,
        guidance_scale: int = 1,
        compile: bool = True,
        default_negative_prompt: str | None = None,
    ):
        self.pipe = LightX2VPipeline(
            model_path=model_path,
            model_cls=model_cls,
            task=generation_type,
        )

        self.pipe.enable_quantize(
            dit_quantized=True,
            quant_scheme="fp8-sgl",
        )

        self.pipe.create_generator(
            attn_mode=attention_backend,
            resize_mode="adaptive",
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

    def generate(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        output_path: str,
        image_paths: list[str] = [],
        seed: int = -1,  # -1 for random seed
        negative_prompt: str | None = None,
    ):
        image_paths_str = ",".join(image_paths)

        if aspect_ratio not in self.aspect_ratios:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        if resolution not in self.aspect_ratios[aspect_ratio]:
            raise ValueError(f"Invalid resolution: {resolution}")

        height, width = self.aspect_ratios[aspect_ratio][resolution]

        self.pipe.generate(
            seed=seed,
            image_path=image_paths_str,
            prompt=prompt,
            negative_prompt=negative_prompt or self.default_negative_prompt,
            save_result_path=output_path,
            height=height,
            width=width,
        )


class QwenImageEdit(BaseModel):
    def __init__(self, model_path: str, compile=True):
        super().__init__(
            model_cls="qwen-image-edit-2511",
            generation_type="i2i",
            model_path=model_path,
            compile=compile,
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
        )


class QwenImage(BaseModel):
    def __init__(self, model_path: str, text_encoder=None, vae=None, compile=True):
        if text_encoder and vae:
            QwenImageRunner.load_model = self._create_patched_load_model(
                text_encoder, vae
            )
        super().__init__(
            model_cls="qwen-image-2512",
            generation_type="t2i",
            model_path=model_path,
            compile=compile,
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
    def __init__(self, model_path: str, compile=True):
        super().__init__(
            model_cls="z_image",
            generation_type="t2i",
            model_path=model_path,
            compile=compile,
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
        )


class Wan22_5B(BaseModel):
    def __init__(self, model_path: str, compile=True):
        super().__init__(
            model_cls="wan2.2",
            generation_type="i2v",
            model_path=model_path,
            compile=compile,
            attention_backend="sage_attn2",
            aspect_ratios={
                "16:9": {"480p": (854, 480), "720p": (1280, 720)},
                "9:16": {"480p": (480, 854), "720p": (720, 1280)},
            },
        )
