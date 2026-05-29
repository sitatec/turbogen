import os
import time
from typing import cast
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

from turbogen.utils import load_flash_attention, disable_manual_memory_gc, set_jit_cache_dirs

# ruff: noqa:E402
set_jit_cache_dirs(Path(__file__).parent.resolve() / ".jit_cache")
load_flash_attention()

from cog import BasePredictor, Input, Path as CogPath
import lightx2v.models.runners.qwen_image.qwen_image_runner  # noqa Needed before importing lightx2v models
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.models.lightx2v_models import QwenImageLite
from turbogen.model_downloads import download_qwen_image


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        t = time.perf_counter()
        qwen_image_path = download_qwen_image(offline=True)
        print(f"Downloaded in {time.perf_counter() - t} seconds")

        t2 = time.perf_counter()
        self.qwen_image = QwenImageLite(qwen_image_path)
        print(f"Model loaded in {time.perf_counter() - t2} seconds")

        self.pipeline = GenerationPipeline(models=[self.qwen_image])

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
            ],
        ),
        resolution: str = Input(
            default="1.3K",
            choices=["1.3K"],
            description="Currently, only the native 1.3k (1328x1328 for square AR) is supported. This is the recommended resolution by the Qwen team for the best results.",
        ),
        seed: int = Input(
            description="Random seed. Set to -1 for random.",
            default=-1,
        ),
    ) -> CogPath:
        model_id = self.qwen_image.model_id
        image_paths = []

        # The lightx2v lib do a lot of torch.cuda.empty_cache() which sync gpu,
        # introducing some latency. So we disable it. TODO: make it configurable
        with disable_manual_memory_gc():
            output_path = self.pipeline.generate(
                model_id=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_paths=image_paths,
                seed=seed,
                resolution=resolution,
                postprocess=False,
                output_dir_path="./output",
            )

        return CogPath(cast(str, output_path))

    def warmup(self) -> None:
        print("Running warmup...")
        self.predict(
            prompt="a cat sitting on a chair",
            aspect_ratio="1:1",
            seed=42,
        )
        print("Warmup complete.")
