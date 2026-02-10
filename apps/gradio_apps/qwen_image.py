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
from core.utils import load_flash_attention_3

load_flash_attention_3()

# ruff: noqa:E402
from model_downloads import download_qwen_image, download_prompt_enhancer
from core.models.lightx2v_models import QwenImageLite
from core.generation_pipeline import GenerationPipeline
from core.services.prompt_enhancer import PromptEnhancer


qwen_image_path = download_qwen_image(te_quant_method="bnb")
prompt_enhancer_path = download_prompt_enhancer(quant_method="bnb")

qwen_image = QwenImageLite(qwen_image_path, rope_type="torch")

pipeline = GenerationPipeline(
    models=[qwen_image],
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=False,
    prompt_enhancing_supported=True,
    title="""
        # 🎨 Qwen Image Generation
        Lightning fast image generation with the latest Qwen Image model (2512)
        """,
)

if __name__ == "__main__":
    app.launch()
