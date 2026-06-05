print("Container Starting...")
import os
import time
from typing import cast

print("Importing COG...")
from cog import BaseRunner, Input, Path


app_root_dir = Path(__file__).parent.resolve()

print("Importing turbogen.utils...")
from turbogen.utils import disable_manual_memory_gc, set_jit_cache_dirs

print("Setting JIT cache dirs...")
# ruff: noqa:E402
jit_cache_dir_root = app_root_dir / ".jit_cache"
set_jit_cache_dirs(jit_cache_dir_root)
print(f"JIT cache dirs set with root: {jit_cache_dir_root}")
try:
    print(f"cache dir content: {os.listdir(jit_cache_dir_root)}")
except Exception as e:
    print(f"Failed to list cache dir content: {e}")

print("Importing lightx2v...")
import lightx2v.models.runners.z_image.z_image_runner  # noqa Needed before importing lightx2v models

print("Importing turbogen.generation_pipeline...")
from turbogen.generation_pipeline import GenerationPipeline

print("Importing turbogen.models.lightx2v_models...")
from turbogen.models.lightx2v_models import ZImageTurbo

print("Importing downloads...")
from downloads import download_zimage_models  # type:ignore

print("All Imports completed")


class Model(BaseRunner):
    # pyrefly: ignore
    def setup(self) -> None:
        print("Setup starting...")

        t = time.perf_counter()
        print("Downloading z_image_models...")
        zimage_path = download_zimage_models()
        print(f"Downloaded in {time.perf_counter() - t} seconds. Path: {zimage_path}")

        t2 = time.perf_counter()
        print("Loading ZImageTurbo...")
        self.zimage = ZImageTurbo(
            model_path=zimage_path,
            quant_scheme="fp8-sgl",
        )
        print(f"Model loaded in {time.perf_counter() - t2} seconds")

        print("Instantiating GenerationPipeline...")
        self.pipeline = GenerationPipeline(models=[self.zimage])

        print(f"Completed setup in {time.perf_counter() - t} seconds")

    # pyrefly: ignore
    def run(
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
    ) -> Path:
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
                output_dir_path="output",
            )

        return Path(cast(str, output_path))

    def warmup(self) -> None:
        print("Running warmup...")
        self.predict(
            prompt="a cat sitting on a chair",
            aspect_ratio="1:1",
            resolution="1K",
            seed=42,
        )
        print("Warmup complete.")
