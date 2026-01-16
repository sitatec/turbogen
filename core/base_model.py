from enum import Enum

import torch


class GenerationType(Enum):
    T2I = "t2i"
    I2I = "i2i"
    I2V = "i2v"
    T2V = "t2v"

    @property
    def is_video(self) -> bool:
        return self in [GenerationType.I2V, GenerationType.T2V]

    @property
    def is_image(self) -> bool:
        return not self.is_video


class BaseModel:
    model_id: str
    generation_type: GenerationType

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
        guidance_scale: int | None = None,
        duration_seconds: float | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError()
