# gradio_ui_factory must be imported first to init ZeroGPU when importing `spaces`
from turbogen.utils.gradio_ui_factory import create_gradio_app
from turbogen.utils import load_sage_attention

load_sage_attention()

# ruff: noqa:E402
from turbogen.model_downloads import download_wan22_models, download_prompt_enhancer
from turbogen.models.lightx2v_models import Wan22Lite
from turbogen.models.base_model import GenerationType
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.services.prompt_enhancer import PromptEnhancer


wan22_i2v_path, wan22_t2v_path = download_wan22_models()
prompt_enhancer_path = download_prompt_enhancer(quant_method="bnb")

wan22_i2v = Wan22Lite(
    wan22_i2v_path,
    generation_type=GenerationType.I2V,
    rope_type="torch",
)
wan22_t2v = Wan22Lite(
    wan22_t2v_path,
    generation_type=GenerationType.T2V,
    rope_type="torch",
)

pipeline = GenerationPipeline(
    models=[wan22_i2v, wan22_t2v],
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=False,
    prompt_enhancing_supported=True,
    title="""
        # 🎨 Wan 2.2 A14B Image-To-Video & Text-To-Video
        Create stunning videos in a flash
        """,
)

if __name__ == "__main__":
    app.launch()
