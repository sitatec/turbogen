from pathlib import Path
from typing import Literal

import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
from core.utils import is_hopper_gpu
from core.models.base_model import BaseModel, GenerationType


class _BaseDiffusersModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        model_name: str,
        model_path: Path,
        generation_type: GenerationType,
        aspect_ratios: dict[str, dict[str, tuple[int, int]]],
        default_negative_prompt: str | None = None,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        infer_steps: int = 4,
        guidance_scale: float = 1,
        quant_scheme: Literal["float8dq_e4m3"] | None = None,
        **kwargs,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.generation_type = generation_type
        self.supported_aspect_ratios = aspect_ratios
        self.default_negative_prompt = default_negative_prompt
        self.default_inference_steps = infer_steps
        self.default_guidance_scale = guidance_scale

        quantization_config = (
            PipelineQuantizationConfig(
                quant_mapping={"transformer": TorchAoConfig(quant_scheme)}
            )
            if quant_scheme
            else None
        )

        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        self.pipe.transformer.set_attention_backend(
            "_flash_3_hub" if is_hopper_gpu() else "sage_hub"
        )

        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()

        if compile:
            self.pipe.transformer.compile_repeated_blocks()

    def generate(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        image_paths: list[str] = [],
        last_frame_path: str | None = None,
        seed: int = -1,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        duration_seconds: float | None = None,
    ) -> torch.Tensor:
        if seed == -1:
            seed = torch.random.initial_seed()
        generator = torch.Generator(device=self.pipe.device).manual_seed(int(seed))

        if aspect_ratio not in self.supported_aspect_ratios:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        if resolution not in self.supported_aspect_ratios[aspect_ratio]:
            raise ValueError(f"Invalid resolution: {resolution}")

        width, height = self.supported_aspect_ratios[aspect_ratio][resolution]

        if image_paths:
            pass

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or self.default_negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps or self.default_inference_steps,
            guidance_scale=guidance_scale
            if guidance_scale is not None
            else self.default_guidance_scale,
            generator=generator,
            output_type="pt",
        )

        if hasattr(output, "frames"):
            return output.frames[0]  # Video pipeline return
        if hasattr(output, "images"):
            return output.images[0]  # Image pipeline return

        return output[0]


class QwenImageEditLite(_BaseDiffusersModel):
    def __init__(
        self,
        model_path: Path,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        lora_configs: list[dict] = [],
        infer_steps: int = 4,
        **kwargs,
    ):
        model_path_str = str(model_path)
        # Add the specific LoRA mentioned in original file
        default_lora = {
            "path": f"{model_path_str}/lora/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "strength": 1,
        }

        super().__init__(
            model_id="k",
            model_name="Qwen Image Edit 2511",
            model_path=model_path,
            generation_type=GenerationType.I2I,
            compile=compile,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=infer_steps,
            lora_configs=[default_lora, *lora_configs],
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


class QwenImageLite(_BaseDiffusersModel):
    def __init__(
        self,
        model_path: Path,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        lora_configs: list[dict] = [],
        infer_steps: int = 4,
        **kwargs,
    ):
        model_path_str = str(model_path)
        default_lora = {
            "path": f"{model_path_str}/lora/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
            "strength": 1,
        }
        super().__init__(
            model_id="l",
            model_name="Qwen Image 2512",
            model_path=model_path,
            generation_type=GenerationType.T2I,
            compile=compile,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=infer_steps,
            lora_configs=[default_lora, *lora_configs],
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


class ZImageTurbo(_BaseDiffusersModel):
    def __init__(
        self,
        model_path: Path,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        lora_configs: list[dict] = [],
        infer_steps: int = 4,
        **kwargs,
    ):
        super().__init__(
            model_id="0",
            model_name="Z-Image-Turbo",
            model_path=model_path,
            generation_type=GenerationType.T2I,
            compile=compile,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=infer_steps,
            lora_configs=lora_configs,
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


class Wan22Lite(_BaseDiffusersModel):
    def __init__(
        self,
        model_path: Path,
        generation_type: Literal[
            GenerationType.I2V, GenerationType.T2V
        ] = GenerationType.I2V,
        compile: bool = False,
        enable_cpu_offload: bool = False,
        lora_configs: list[dict] = [],
        infer_steps: int = 4,
        **kwargs,
    ):
        super().__init__(
            model_id="m",
            model_name=f"Wan 2.2 A14B {generation_type.value.upper()}",
            model_path=model_path,
            generation_type=generation_type,
            compile=compile,
            enable_cpu_offload=enable_cpu_offload,
            infer_steps=infer_steps,
            lora_configs=lora_configs,
            aspect_ratios={
                "16:9": {"480p": (854, 480), "720p": (1280, 720)},
                "9:16": {"480p": (480, 854), "720p": (720, 1280)},
            },
            **kwargs,
        )
