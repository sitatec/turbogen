import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            parent for parent in Path(__file__).parents if parent.name == "turbogen"
        ).resolve()
    ),
)

from apps.gradio_apps.ui_factory import (
    create_gradio_app,  # must be imported first to init ZeroGPU when importing `spaces`
)
from model_downloads import (
    download_zimage_models,
    download_image_scorer,
    download_prompt_enhancer,
)
from core.models.lightx2v_models import ZImageTurbo
from core.generation_pipeline import GenerationPipeline
from core.services.media_scoring.image_scorer import ImageScorer
from core.services.prompt_enhancer import PromptEnhancer
from core.services.nsfw_detector import NsfwDetector


zimage_turbo_path = download_zimage_models()
image_scorer_path = download_image_scorer()
prompt_enhancer_path = download_prompt_enhancer()

zimage_turbo = ZImageTurbo(zimage_turbo_path)

pipeline = GenerationPipeline(
    models=[zimage_turbo],
    nsfw_detector=NsfwDetector(),
    image_scorer=ImageScorer(image_scorer_path),
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=True,
    title="""
        # 🎨 Z-Image 
        Lightning fast image generation with the Z-Image models
        """,
)

if __name__ == "__main__":
    app.launch()
