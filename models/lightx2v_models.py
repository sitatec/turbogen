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
        generation_type: Literal["t2i", "i2i"],
        fp8_weights_path: str,
        aspect_ratios: dict[str, dict[str, tuple[int, int]]],
        model_path: str = "",
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
            dit_quantized_ckpt=fp8_weights_path,
            quant_scheme="fp8-sgl",
        )

        self.pipe.create_generator(
            attn_mode="flash_attn3",
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
        image_paths: list[str],
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        output_path: str,
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
    def __init__(self):
        super().__init__(
            model_cls="qwen-image-edit-2511",
            generation_type="i2i",
            fp8_weights_path="lightx2v/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
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
    def __init__(self):
        super().__init__(
            model_cls="qwen-image-2512",
            generation_type="t2i",
            fp8_weights_path="lightx2v/Qwen-Image-2512-Lightning/qwen_image_2512_fp8_e4m3fn_scaled_4steps_v1.0.safetensors",
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


class ZImageTurbo(BaseModel):
    def __init__(self):
        super().__init__(
            model_cls="z_image",
            generation_type="t2i",
            fp8_weights_path="TODO",
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
