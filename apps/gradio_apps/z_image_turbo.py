# gradio_ui_factory must be imported first to init ZeroGPU when importing `spaces`
from turbogen.utils.gradio_ui_factory import create_gradio_app
from turbogen.utils import load_flash_attention_3

load_flash_attention_3()

# ruff: noqa:E402
from turbogen.model_downloads import (
    download_zimage_models,
    download_image_scorer,
    download_prompt_enhancer,
    download_nsfw_model,
)
from turbogen.models.lightx2v_models import ZImageTurbo
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.services.media_scoring.image_scorer import ImageScorer
from turbogen.services.prompt_enhancer import PromptEnhancer
from turbogen.services.nsfw_detector import NsfwDetector


zimage_turbo_path = download_zimage_models()
image_scorer_path = download_image_scorer()
prompt_enhancer_path = download_prompt_enhancer()
nsfw_model_path = download_nsfw_model()

zimage_turbo = ZImageTurbo(zimage_turbo_path, rope_type="torch")

pipeline = GenerationPipeline(
    models=[zimage_turbo],
    nsfw_detector=NsfwDetector(nsfw_model_path),
    image_scorer=ImageScorer(image_scorer_path),
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=True,
    prompt_enhancing_supported=True,
    title="""
        # 🎨 Z-Image 
        Lightning fast image generation with the Z-Image models
        """,
)

if __name__ == "__main__":
    app.launch()
