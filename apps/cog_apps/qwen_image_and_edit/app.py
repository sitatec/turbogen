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
from core.models import QwenImageLite, QwenImageEditLite
from model_downloads import download_qwen_models


class Model(BasePredictor):
    # pyrefly: ignore
    def setup(self) -> None:
        qwen_image_edit_path, qwen_image_path = download_qwen_models()

        self.qwen_image_edit = QwenImageEditLite(qwen_image_edit_path)
        self.qwen_image = QwenImageLite(
            qwen_image_path,
            vae=self.qwen_image_edit.pipe.runner.vae,
            text_encoder=self.qwen_image_edit.pipe.runner.text_encoders,
        )

        self.pipeline = GenerationPipeline(
            models=[self.qwen_image, self.qwen_image_edit]
        )

    # pyrefly: ignore
    def predict(
        self,
        prompt: str = Input(description="Description of the image"),
        images: list[CogPath] = Input(
            description="Input image for image editing with. If provided, the edit model will be used.",
            default=None,
        ),
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
    ) -> CogPath:
        t = time.perf_counter()

        if images:
            model_id = self.qwen_image_edit.model_id
            resolution = "1K"
            image_paths = [str(image) for image in images]
            print(
                f"Generating with model={self.qwen_image_edit.model_name}, aspect_ratio={aspect_ratio}..."
            )
        else:
            model_id = self.qwen_image.model_id
            resolution = "1.3K"
            image_paths = []
            print(
                f"Generating with model={self.qwen_image.model_name}, aspect_ratio={aspect_ratio}..."
            )

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

        print(f"Generated in {time.perf_counter() - t} seconds")
        return CogPath(cast(str, output_path))
