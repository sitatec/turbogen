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

from cog import BasePredictor, Input, Path as CogPath
from core.generation_pipeline import GenerationPipeline
from core.models.base_model import GenerationType
from core.models import Wan22Lite
from model_downloads import download_wan22_models


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        wan22_i2v_path, wan22_t2v_path = download_wan22_models()

        self.wan22_t2v = Wan22Lite(
            model_path=wan22_t2v_path, generation_type=GenerationType.T2V
        )
        self.wan22_i2v = Wan22Lite(
            model_path=wan22_i2v_path, generation_type=GenerationType.I2V
        )

        self.pipeline = GenerationPipeline(
            models=[self.wan22_t2v, self.wan22_i2v],
            nsfw_detector=None,
            image_scorer=None,
            video_scorer=None,
        )

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

        if image:
            model_id = self.wan22_i2v.model_id
            image_paths = [str(image)]
            print(
                f"Generating with model={self.wan22_i2v.model_name}, resolution={resolution}, aspect_ratio={aspect_ratio}..."
            )
        else:
            model_id = self.wan22_t2v.model_id
            image_paths = []
            print(
                f"Generating with model={self.wan22_t2v.model_name}, resolution={resolution}, aspect_ratio={aspect_ratio}..."
            )

        output_path = self.pipeline.generate(
            model_id=model_id,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_paths=image_paths,
            last_frame_path=str(last_frame) if last_frame else None,
            seed=seed,
            resolution=resolution,
            postprocess=False,
            output_dir_path="./output",
        )

        print(f"Generated in {time.perf_counter() - t} seconds")
        return CogPath(cast(str, output_path))
