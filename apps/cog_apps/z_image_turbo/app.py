import time
from typing import cast
from pathlib import Path

from turbogen.utils import load_flash_attention_3, disable_manual_memory_gc, set_jit_cache_dirs

# ruff: noqa:E402
set_jit_cache_dirs(Path(__file__).parent.resolve() / ".jit_cache")
load_flash_attention_3()

from cog import BasePredictor, Input, Path as CogPath
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.models.lightx2v_models import ZImageTurbo
from turbogen.model_downloads import download_zimage_models


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        t = time.perf_counter()
        zimage_path = download_zimage_models()
        print(f"Downloaded in {time.perf_counter() - t} seconds")

        t2 = time.perf_counter()
        self.zimage = ZImageTurbo(model_path=zimage_path)
        print(f"Model loaded in {time.perf_counter() - t2} seconds")

        self.pipeline = GenerationPipeline(models=[self.zimage])

        print(f"Completed setup in {time.perf_counter() - t} seconds")

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

        # The lightx2v lib do a lot of torch.cuda.empty_cache() which sync gpu,
        # introducing some latency. So we disable it. TODO: make it configurable
        with disable_manual_memory_gc():
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

    def warmup(self) -> None:
        print("Running warmup...")
        self.predict(
            prompt="a cat sitting on a chair",
            aspect_ratio="1:1",
            resolution="1K",
            seed=42,
        )
        print("Warmup complete.")
