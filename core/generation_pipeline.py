from __future__ import annotations

import gc
from typing import TYPE_CHECKING
from tempfile import mkdtemp
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from core.models.base_model import GenerationType
from core.services.nsfw_detector import NsfwLevel
from core.utils.video_utils import save_video_tensor
from core.utils.image_utils import (
    create_exif_data,
    image_tensor_to_pil,
    image_tensor_to_numpy,
    convert_to_webp_with_metadata,
    generate_thumbhash,
)

if TYPE_CHECKING:
    from core.models.base_model import BaseModel
    from core.services.nsfw_detector import NsfwDetector
    from core.services.media_scoring import ImageScorer, VideoScorer
    from core.services.prompt_enhancer import PromptEnhancer


@dataclass
class ProcessedOutput:
    generated_media_path: str
    thumbnail_path: str
    thumbhash: str
    nsfwLevel: NsfwLevel
    quality_score: float


class GenerationPipeline:
    def __init__(
        self,
        models: list[BaseModel],
        prompt_enhancer: PromptEnhancer | None = None,
        nsfw_detector: NsfwDetector | None = None,
        image_scorer: ImageScorer | None = None,
        video_scorer: VideoScorer | None = None,
    ):
        assert len(models) > 0, "The models argument cannot be empty"

        self.models = models
        self.nsfw_detector = nsfw_detector
        self.image_scorer = image_scorer
        self.video_scorer = video_scorer
        self.prompt_enhancer = prompt_enhancer

    @torch.inference_mode()
    def generate(
        self,
        model_id: str,
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
        postprocess: bool = False,
        enhance_prompt: bool = False,
        output_dir_path: str | None = None,
        metadata: dict | None = None,
    ) -> ProcessedOutput | str:
        assert (
            not postprocess
            or self.nsfw_detector
            and (self.video_scorer or self.image_scorer)
        ), "NSFW detector and at least one media scorer are required for postprocessing"

        model = next(
            (model for model in self.models if model.model_id == model_id), None
        )
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        if enhance_prompt:
            prompt = self._enhance_prompt(
                prompt,
                model.generation_type,
                image_paths,
                last_frame_path,
            )
            if metadata:
                metadata["enhanced_prompt"] = prompt

        output = model.generate(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_paths=image_paths,
            last_frame_path=last_frame_path,
            seed=seed,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            duration_seconds=duration_seconds,
        )
        self._free_memory()

        output_dir_path = output_dir_path or mkdtemp()
        if postprocess:
            result = self._process_and_save_output(
                output, model.generation_type, output_dir_path, metadata
            )
        else:
            result = self._save_output(
                output, model.generation_type, output_dir_path, metadata
            )

        self._free_memory()
        return result

    def _enhance_prompt(
        self,
        prompt: str,
        generation_type: GenerationType,
        image_paths: list[str],
        last_frame_path: str | None,
    ):
        assert self.prompt_enhancer is not None, "Prompt Enhancer not initialized"
        input_images = image_paths

        if generation_type == GenerationType.I2V:
            assert len(input_images) == 1, (
                "Only 1 input image is supported for I2V, if you want to provide a last frame, use the last_frame_path instead"
            )
            if last_frame_path:
                input_images.append(last_frame_path)

        return self.prompt_enhancer.enhance_prompt(
            prompt, generation_type, input_images
        )

    def _process_and_save_output(
        self,
        output: torch.Tensor,
        generation_type: GenerationType,
        output_dir_path: str,
        metadata: dict | None = None,
        fps: int = 16,
        thumbnail_quality: int = 90,
    ) -> ProcessedOutput:
        if generation_type.is_video:
            assert self.video_scorer is not None
            output_path = f"{output_dir_path}/output.mp4"
            thumbnail_path = f"{output_dir_path}/thumbnail.webp"

            first_frame = image_tensor_to_numpy(output[0])
            thumbnail = self._create_thumbnail(first_frame)
            thumbhash = generate_thumbhash(thumbnail)
            quality_score = self.video_scorer.score(output)[0]

            save_video_tensor(output, output_path, fps)
            convert_to_webp_with_metadata(
                thumbnail,
                metadata,
                output_path=thumbnail_path,
                quality=thumbnail_quality,
            )
        else:
            assert self.image_scorer is not None
            output_path = f"{output_dir_path}/output.webp"
            thumbnail_path = f"{output_dir_path}/thumbnail.webp"

            image = image_tensor_to_numpy(output)
            thumbnail = self._create_thumbnail(image)
            thumbhash = generate_thumbhash(thumbnail)
            quality_score = self.image_scorer.score(output)

            convert_to_webp_with_metadata(image, metadata, output_path=output_path)
            convert_to_webp_with_metadata(
                thumbnail,
                metadata,
                output_path=thumbnail_path,
                quality=thumbnail_quality,
            )

        return ProcessedOutput(
            generated_media_path=output_path,
            thumbnail_path=thumbnail_path,
            thumbhash=thumbhash,
            nsfwLevel=self._get_nsfw_level(output, generation_type),
            quality_score=quality_score,
        )

    def _save_output(
        self,
        output: torch.Tensor,
        generation_type: GenerationType,
        output_dir_path: str,
        metadata: dict | None = None,
        fps: int = 16,
    ) -> str:
        output_path = f"{output_dir_path}/output.webp"
        if generation_type.is_image:
            image_tensor_to_pil(output).save(
                output_path,
                format="WEBP",
                quality=100,
                exif=create_exif_data(metadata) if metadata else None,
            )
        else:
            output_path = f"{output_dir_path}/output.mp4"
            save_video_tensor(output, output_path, fps)

        return output_path

    def _create_thumbnail(
        self,
        img: np.ndarray,
        target_width: int = 480,
        min_height: int = 360,
    ):
        """Resize to [target_width] while preventing the new height from being smaller than [min_height]"""
        height, width = img.shape[:2]
        scale = max(target_width / width, min_height / height)

        if scale >= 1.0:  # Avoid upscaling
            return img

        new_w = int(round(width * scale))
        new_h = int(round(height * scale))

        # If source is much larger, downscale to 1.5x target using INTER_AREA as a first pass.
        # This removes high-frequency noise and prevents Lanczos "ringing" (halos).
        if scale < 0.6:
            img = cv2.resize(
                img,
                (int(new_w * 1.5), int(new_h * 1.5)),
                interpolation=cv2.INTER_AREA,
            )

        # Final low scale resize to target using Lanczos for nice acuity
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _get_nsfw_level(
        self,
        output: torch.Tensor,
        generation_type: GenerationType,
        fps=16,
    ) -> NsfwLevel:
        assert self.nsfw_detector, "NSFW detector is not initialized"

        if generation_type in [GenerationType.T2V, GenerationType.I2V]:
            frames = self._selected_video_frames(output, fps)
            if not frames:
                return NsfwLevel.SAFE

            levels = self.nsfw_detector.get_nsfw_level(frames)
            if not isinstance(levels, list):
                levels = [levels]

            return max(levels, key=lambda level: level.rank)
        else:
            result = self.nsfw_detector.get_nsfw_level(output)
            return result if not isinstance(result, list) else result[0]

    def _selected_video_frames(
        self,
        output: torch.Tensor,
        fps: int,
    ) -> torch.Tensor:
        assert output.dim() == 4

        num_frames = output.shape[0]
        video_duration = num_frames / fps

        if video_duration >= 60:
            duration_interval = 4
        elif video_duration >= 30:
            duration_interval = 3
        else:
            duration_interval = 2

        interval = int(duration_interval * fps)
        indices = slice(0, num_frames, interval)
        return output[indices]

    def _free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()


__all__ = [ProcessedOutput, GenerationPipeline]
