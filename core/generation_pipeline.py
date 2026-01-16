from tempfile import mkdtemp
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from core.base_model import BaseModel, GenerationType
from core.services.nsfw_detector import NsfwDetector, NsfwLevel
from core.utils.video_utils import save_video_tensor
from core.utils.image_utils import (
    create_exif_data,
    image_tensor_to_pil,
    image_tensor_to_numpy,
    convert_to_webp_with_metadata,
    generate_thumbhash,
)


@dataclass
class ProcessedOutput:
    generated_media_path: str
    thumbnail_path: str
    thumbhash: str
    nsfwLevel: NsfwLevel | None


class GenerationPipeline:
    def __init__(
        self,
        models: list[BaseModel],
        nsfw_detector: NsfwDetector | None,
    ):
        assert len(models) > 0, "The models argument cannot be empty"

        self.models = models
        self.nsfw_detector = nsfw_detector

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
        guidance_scale: int | None = None,
        duration_seconds: float | None = None,
        postprocess: bool = True,
        output_dir_path: str | None = None,
        metadata: dict | None = None,
    ) -> ProcessedOutput | str:
        assert not postprocess or self.nsfw_detector, (
            "NSFW detector is required for postprocessing"
        )

        model = next(
            (model for model in self.models if model.model_id == model_id), None
        )
        if model is None:
            raise ValueError(f"Model {model_id} not found")

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

        output_dir_path = output_dir_path or mkdtemp()
        if postprocess:
            return self._save_processed_output(
                output, model.generation_type, output_dir_path, metadata
            )

        return self._save_output(
            output, model.generation_type, output_dir_path, metadata
        )

    def _save_processed_output(
        self,
        output: torch.Tensor,
        generation_type: GenerationType,
        output_dir_path: str,
        metadata: dict | None = None,
        fps: int = 16,
        thumbnail_quality: int = 90,
    ) -> ProcessedOutput:
        if generation_type.is_video:
            output_path = f"{output_dir_path}/output.mp4"
            thumbnail_path = f"{output_dir_path}/thumbnail.webp"

            first_frame = image_tensor_to_numpy(output[0])
            thumbnail = self._create_thumbnail(first_frame)
            thumbhash = generate_thumbhash(thumbnail)

            save_video_tensor(output, output_path, fps)
            convert_to_webp_with_metadata(
                thumbnail,
                metadata,
                output_path=thumbnail_path,
                quality=thumbnail_quality,
            )
        else:
            output_path = f"{output_dir_path}/output.webp"
            thumbnail_path = f"{output_dir_path}/thumbnail.webp"

            image = image_tensor_to_numpy(output)
            thumbnail = self._create_thumbnail(image)
            thumbhash = generate_thumbhash(thumbnail)

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
        target_width: int = 512,
        min_height: int = 400,
    ):
        height, width = img.shape[:2]

        scale = max(target_width / width, min_height / height)

        new_w = int(round(width * scale))
        new_h = int(round(height * scale))

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
        duration_interval: float = 2,
    ) -> torch.Tensor:
        assert output.dim() == 4

        num_frames = output.shape[0]
        interval = int(duration_interval * fps)
        indices = range(0, num_frames, interval)
        return torch.tensor([output[:, i, :, :] for i in indices])
