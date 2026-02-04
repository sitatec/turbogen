import sys
import time
from typing import cast
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            parent for parent in Path(__file__).parents if parent.name == "turbogen"
        ).resolve()
    ),
)

from core.utils import load_flash_attention_3

# ruff: noqa:E402
load_flash_attention_3()

from cog import BasePredictor, Input, Path as CogPath
from core.generation_pipeline import GenerationPipeline
from core.models.lightx2v_models import ZImageTurbo
from model_downloads import download_zimage_models


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        zimage_path = download_zimage_models()

        self.zimage = ZImageTurbo(model_path=zimage_path)

        self.pipeline = GenerationPipeline(models=[self.zimage])

    # pyrefly: ignore
    def predict(
        self,
        prompt: str = Input(description="Description of the image"),
        aspect_ratio: str = Input(
            default="1:1",
            choices=[
                "1:1",
                "16:9",
                "9:16",
                "4:3",
                "3:4",
                "3:2",
                "2:3",
                "5:4",
                "4:5",
                "21:9",
                "9:21",
            ],
        ),
        resolution: str = Input(
            default="1.3K",
            choices=["1K", "1.3K", "1.5K"],
        ),
        seed: int = Input(
            description="Random seed. Set to -1 for random.",
            default=-1,
        ),
    ) -> CogPath:
        t = time.perf_counter()

        output_path = self.pipeline.generate(
            model_id=self.zimage.model_id,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            resolution=resolution,
            postprocess=False,
            output_dir_path="./output",
        )

        print(f"Generated in {time.perf_counter() - t} seconds")
        return CogPath(cast(str, output_path))
