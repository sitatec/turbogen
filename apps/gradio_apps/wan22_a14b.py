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
    download_wan22_models,
    download_video_scorer,
    download_prompt_enhancer,
)
from core.models.lightx2v_models import Wan22Lite
from core.models.base_model import GenerationType
from core.generation_pipeline import GenerationPipeline
from core.services.media_scoring.video_scorer import VideoScorer
from core.services.prompt_enhancer import PromptEnhancer
from core.services.nsfw_detector import NsfwDetector


wan22_i2v_path, wan22_t2v_path = download_wan22_models()
video_scorer_path = download_video_scorer()
prompt_enhancer_path = download_prompt_enhancer()

wan22_i2v = Wan22Lite(wan22_i2v_path, generation_type=GenerationType.I2V)
wan22_t2v = Wan22Lite(wan22_t2v_path, generation_type=GenerationType.T2V)

pipeline = GenerationPipeline(
    models=[wan22_i2v, wan22_t2v],
    nsfw_detector=NsfwDetector(),
    video_scorer=VideoScorer(video_scorer_path),
    prompt_enhancer=PromptEnhancer(prompt_enhancer_path),
)

app = create_gradio_app(
    pipeline,
    postprocessing_supported=True,
    title="""
        # 🎨 Wan 2.2 A14B Image-To-Video & Text-To-Video
        Create stunning videos in a flash
        """,
)

if __name__ == "__main__":
    app.launch()
