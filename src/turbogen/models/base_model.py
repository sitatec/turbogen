from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import Tensor


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
    model_name: str
    generation_type: GenerationType
    supported_aspect_ratios: dict[str, dict[str, tuple[int, int]]]
    default_negative_prompt: str | None
    default_inference_steps: int
    default_guidance_scale: float

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
    ) -> Tensor:
        raise NotImplementedError()
