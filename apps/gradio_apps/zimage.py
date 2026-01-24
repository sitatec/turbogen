import sys

sys.path.insert(0, ".")

from model_downloads import download_zimage_models, download_image_scorer
from models.lightx2v_models import ZImageTurbo
from core.generation_pipeline import GenerationPipeline
from core.services.media_scoring.image_scorer import ImageScorer
from core.services.nsfw_detector import NsfwDetector
from apps.gradio_apps.ui_factory import create_gradio_app


if __name__ == "__main__":
    zimage_turbo_path = download_zimage_models()
    image_scorer_path = download_image_scorer()

    zimage_turbo = ZImageTurbo(zimage_turbo_path)

    pipeline = GenerationPipeline(
        models=[zimage_turbo],
        nsfw_detector=NsfwDetector(),
        image_scorer=ImageScorer(image_scorer_path),
        video_scorer=None,
    )

    app = create_gradio_app(
        pipeline,
        postprocessing_supported=True,
        title="""
            # 🎨 Z-Image 
            Lightning fast image generation with the Z-Image models
            """,
    )
    app.launch()
