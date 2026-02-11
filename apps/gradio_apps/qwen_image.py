# gradio_ui_factory must be imported first to init ZeroGPU when importing `spaces`
from turbogen.utils.gradio_ui_factory import create_gradio_app
from turbogen.utils import load_flash_attention_3

load_flash_attention_3()

# ruff: noqa:E402
from turbogen.model_downloads import (
    download_qwen_image,
    download_image_scorer,
    download_prompt_enhancer,
    download_nsfw_model,
)
from turbogen.models.lightx2v_models import QwenImageLite
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.services.media_scoring.image_scorer import ImageScorer
from turbogen.services.prompt_enhancer import PromptEnhancer
from turbogen.services.nsfw_detector import NsfwDetector


qwen_image_path = download_qwen_image(te_quant_method="bnb")
image_scorer_path = download_image_scorer(quant_method="bnb")
nsfw_model_path = download_nsfw_model()
prompt_enhancer_path = download_prompt_enhancer(quant_method="bnb")

qwen_image = QwenImageLite(qwen_image_path, rope_type="torch")
qwen_image.pipe.runner.vae.model.enable_tiling(
    tile_sample_min_height=768,
    tile_sample_min_width=768,
    tile_sample_stride_height=768 - 96,
    tile_sample_stride_width=768 - 96,
)

pipeline = GenerationPipeline(
    models=[qwen_image],
    nsfw_detector=NsfwDetector(nsfw_model_path),
    image_scorer=ImageScorer(image_scorer_path),
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=True,
    prompt_enhancing_supported=True,
    title="""
        # 🎨 Qwen Image Generation
        Lightning fast image generation with the latest Qwen Image model (2512)
        """,
)

if __name__ == "__main__":
    app.launch()
