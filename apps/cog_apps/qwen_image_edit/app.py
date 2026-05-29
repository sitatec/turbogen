import os
import time
from typing import cast
from cog import BasePredictor, Input, Path

os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

from turbogen.utils import load_flash_attention, disable_manual_memory_gc, set_jit_cache_dirs

# ruff: noqa:E402
set_jit_cache_dirs(Path(__file__).parent.resolve() / ".jit_cache")
load_flash_attention()

import lightx2v.models.runners.qwen_image.qwen_image_runner  # noqa Needed before importing lightx2v models
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.models.lightx2v_models import QwenImageEditLite
from turbogen.model_downloads import download_qwen_image_edit


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        t = time.perf_counter()
        qwen_image_edit_path = download_qwen_image_edit(offline=True)
        print(f"Downloaded in {time.perf_counter() - t} seconds")

        t2 = time.perf_counter()
        self.qwen_image_edit = QwenImageEditLite(qwen_image_edit_path)
        print(f"Model loaded in {time.perf_counter() - t2} seconds")

        self.pipeline = GenerationPipeline(models=[self.qwen_image_edit])

        print(f"Completed setup in {time.perf_counter() - t} seconds")

    # pyrefly: ignore
    def predict(
        self,
        prompt: str = Input(description="Description of the image"),
        images: list[Path] = Input(description="Input image for image editing"),
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
        seed: int = Input(
            description="Random seed. Set to -1 for random.",
            default=-1,
        ),
    ) -> Path:
        # The lightx2v lib do a lot of torch.cuda.empty_cache() which sync gpu,
        # introducing some latency. So we disable it. TODO: make it configurable
        with disable_manual_memory_gc():
            output_path = self.pipeline.generate(
                model_id=self.qwen_image_edit.model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_paths=[str(image) for image in images],
                seed=seed,
                resolution="1K",
                postprocess=False,
                output_dir_path="./output",
            )

        return Path(cast(str, output_path))

    def warmup(self) -> None:
        import tempfile
        from PIL import Image

        print("Running warmup...")
        # Create a small solid-colour image to satisfy the required image input
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            Image.new("RGB", (1024, 1024), color=(128, 128, 128)).save(tmp_f.name)
            self.predict(
                prompt="a cat sitting on a chair",
                images=[Path(tmp_f.name)],
                aspect_ratio="1:1",
                seed=42,
            )
        print("Warmup complete.")
