import time
from typing import cast
from pathlib import Path

from turbogen.utils import load_sage_attention, disable_manual_memory_gc, set_jit_cache_dirs

# ruff: noqa:E402
set_jit_cache_dirs(Path(__file__).parent.resolve() / ".jit_cache")
load_sage_attention()

from cog import BasePredictor, Input, Path as CogPath
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.models.base_model import GenerationType
from turbogen.models.lightx2v_models import Wan22Lite
from turbogen.model_downloads import download_wan22_models


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        t = time.perf_counter()
        wan22_i2v_path, _ = download_wan22_models()
        print(f"Downloaded in {time.perf_counter() - t} seconds")

        t2 = time.perf_counter()
        self.wan22_i2v = Wan22Lite(model_path=wan22_i2v_path, generation_type=GenerationType.I2V)
        print(f"Model loaded in {time.perf_counter() - t2} seconds")

        self.pipeline = GenerationPipeline(models=[self.wan22_i2v])

        print(f"Completed setup in {time.perf_counter() - t} seconds")

    # pyrefly: ignore
    def predict(
        self,
        prompt: str = Input(description="Description of the video"),
        image: CogPath = Input(
            description="Input image for Image-To-Video",
            default=None,
        ),
        aspect_ratio: str = Input(
            default="16:9",
            choices=["16:9", "9:16"],
        ),
        resolution: str = Input(
            default="720p",
            choices=["480p", "720p"],
        ),
        last_frame: CogPath = Input(
            default=None,
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
                model_id=self.wan22_i2v.model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_paths=[str(image)],
                last_frame_path=str(last_frame) if last_frame else None,
                seed=seed,
                resolution=resolution,
                postprocess=False,
                output_dir_path="./output",
            )

        print(f"Generated in {time.perf_counter() - t} seconds")
        return CogPath(cast(str, output_path))

    def warmup(self) -> None:
        import tempfile
        from PIL import Image

        print("Running warmup...")
        # Create a small solid-colour image to satisfy the required image input
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_f:
            Image.new("RGB", (480, 480), color=(128, 128, 128)).save(tmp_f.name)
            self.predict(
                prompt="a cat sitting on a chair",
                image=CogPath(tmp_f.name),
                aspect_ratio="16:9",
                resolution="480p",
                seed=42,
            )
        print("Warmup complete.")
