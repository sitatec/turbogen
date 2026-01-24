import sys

sys.path.insert(0, ".")

from model_downloads import download_wan22_models, download_video_scorer
from models.lightx2v_models import Wan22Lite
from core.base_model import GenerationType
from core.generation_pipeline import GenerationPipeline
from core.services.media_scoring.video_scorer import VideoScorer
from core.services.nsfw_detector import NsfwDetector
from apps.gradio_apps.ui_factory import create_gradio_app


if __name__ == "__main__":
    wan22_i2v_path, wan22_t2v_path = download_wan22_models()
    video_scorer_path = download_video_scorer()

    wan22_i2v = Wan22Lite(wan22_i2v_path, generation_type=GenerationType.I2V)
    wan22_t2v = Wan22Lite(wan22_t2v_path, generation_type=GenerationType.T2V)

    pipeline = GenerationPipeline(
        models=[wan22_i2v, wan22_t2v],
        nsfw_detector=NsfwDetector(),
        image_scorer=None,
        video_scorer=VideoScorer(video_scorer_path),
    )

    app = create_gradio_app(
        pipeline,
        postprocessing_supported=True,
        title="""
            # 🎨 Wan 2.2 A14B Image-To-Video & Text-To-Video
            Create stunning videos in a flash
            """,
    )
    app.launch()
