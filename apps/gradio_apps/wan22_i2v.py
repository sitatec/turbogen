# gradio_ui_factory must be imported first to init ZeroGPU when importing `spaces`
from turbogen.utils.gradio_ui_factory import create_gradio_app
from turbogen.utils import load_sage_attention

load_sage_attention()

# ruff: noqa:E402
from turbogen.model_downloads import (
    download_wan22_models,
    download_prompt_enhancer,
    download_nsfw_model,
    download_video_scorer,
)
from turbogen.models.lightx2v_models import Wan22Lite
from turbogen.models.base_model import GenerationType
from turbogen.generation_pipeline import GenerationPipeline
from turbogen.services.prompt_enhancer import PromptEnhancer
from turbogen.services.media_scoring.video_scorer import VideoScorer
from turbogen.services.nsfw_detector import NsfwDetector

nsfw_model_path = download_nsfw_model()
video_scorer_path = download_video_scorer()
wan22_i2v_path, _ = download_wan22_models()
prompt_enhancer_path = download_prompt_enhancer(quant_method="bnb")

wan22_i2v = Wan22Lite(
    wan22_i2v_path,
    generation_type=GenerationType.I2V,
    rope_type="torch",
)

pipeline = GenerationPipeline(
    models=[wan22_i2v],
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
    nsfw_detector=NsfwDetector(nsfw_model_path),
    video_scorer=VideoScorer(video_scorer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=True,
    prompt_enhancing_supported=True,
    title="""
        # 🎨 Wan 2.2 A14B Image-To-Video
        Create stunning videos in a flash
        """,
)

if __name__ == "__main__":
    app.launch()
