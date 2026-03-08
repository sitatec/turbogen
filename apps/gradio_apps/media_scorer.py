import spaces
import gradio as gr
import json
import shutil
import uuid
import aiohttp
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import decord

from turbogen.utils import load_flash_attention_3, create_thumbnail

load_flash_attention_3()

# ruff: noqa:E402
from turbogen.model_downloads import (
    download_video_scorer,
    download_image_scorer,
    download_nsfw_model,
)
from turbogen.services.media_scoring.image_scorer import ImageScorer
from turbogen.services.media_scoring.video_scorer import VideoScorer
from turbogen.services.nsfw_detector import NsfwDetector
from turbogen.utils.image_utils import generate_thumbhash, convert_to_webp_with_metadata

video_scorer_path = download_video_scorer()
image_scorer_path = download_image_scorer(quant_method="bnb")
nsfw_model_path = download_nsfw_model()

image_scorer = ImageScorer(image_scorer_path)
video_scorer = VideoScorer(video_scorer_path)
nsfw_detector = NsfwDetector(nsfw_model_path)

REQUESTS_DIR = Path("/tmp/media_scorer_requests")
REQUESTS_DIR.mkdir(parents=True, exist_ok=True)


async def download_file(session: aiohttp.ClientSession, url: str, output_path: str):
    """Download a single file from URL asynchronously."""
    async with session.get(url.strip()) as response:
        response.raise_for_status()
        content = await response.read()
        with open(output_path, "wb") as f:
            f.write(content)
        return output_path


@spaces.GPU(duration=15)
def evaluate_image(image: torch.Tensor):
    quality_score = image_scorer.score(image)
    nsfw_levels = nsfw_detector.get_nsfw_level(image)
    nsfw_level = nsfw_levels.value if not isinstance(nsfw_levels, list) else nsfw_levels[0].value
    return quality_score, nsfw_level


@spaces.GPU(duration=30)
def evaluate_video(video_tensor: torch.Tensor, fps: float):
    # We limit the fps to max 30 to reduce memory processing time. But since this is for quality scoring,
    # videos with fps<=30 and fps>30 would be scored unfairly since the later got its native fps reduced.
    # So by default we divide by 2 to reduce the rate of unfair scoring. The scoring model was trained on low fps so, this will work.
    scoring_fps = max(30, round(fps / 2))
    quality_score = video_scorer.score(video_tensor, fps=scoring_fps)[0]
    nsfw_level = nsfw_detector.get_video_nsfw_level(video_tensor, fps=int(fps))
    return quality_score, nsfw_level


async def process_image(input_mode, file_path, url, metadata_input):
    request_id = str(uuid.uuid4())
    request_dir = REQUESTS_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    try:
        if input_mode == "URL":
            if not url:
                raise gr.Error("URL is required")
            image_path = str(request_dir / "input_image.jpg")
            async with aiohttp.ClientSession() as session:
                await download_file(session, url, image_path)
        else:
            if not file_path:
                raise gr.Error("Upload is required")
            # Copy to request_dir to keep it together
            image_path = str(request_dir / Path(file_path).name)
            shutil.copy(file_path, image_path)

        np_image = np.array(Image.open(image_path).convert("RGB"))
        quality_score, nsfw_level = evaluate_image(torch.from_numpy(np_image))

        thumbnail = create_thumbnail(np_image)
        thumbhash = generate_thumbhash(thumbnail)

        metadata = None
        if metadata_input:
            try:
                metadata = json.loads(metadata_input)
            except Exception as e:
                print(f"Failed to parse metadata: {e}")

        thumb_path = str(request_dir / "thumbnail.webp")
        convert_to_webp_with_metadata(thumbnail, metadata, quality=90, output_path=thumb_path)

        return thumb_path, str(nsfw_level), quality_score, thumbhash
    except Exception as err:
        raise gr.Error(f"Generation failed: {err}") from err
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


async def process_video(input_mode, file_path, url, metadata_input):
    request_id = str(uuid.uuid4())
    request_dir = REQUESTS_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    try:
        if input_mode == "URL":
            if not url:
                raise gr.Error("URL is required")
            video_path = str(request_dir / "input_video.mp4")
            async with aiohttp.ClientSession() as session:
                await download_file(session, url, video_path)
        else:
            if not file_path:
                raise gr.Error("Upload is required")
            video_path = str(request_dir / Path(file_path).name)
            shutil.copy(file_path, video_path)

        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        frames = vr.get_batch(range(len(vr))).asnumpy()
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        first_frame = frames[0]

        quality_score, nsfw_level = evaluate_video(video_tensor, fps)

        thumbnail = create_thumbnail(first_frame)
        thumbhash = generate_thumbhash(thumbnail)

        metadata = None
        if metadata_input:
            try:
                metadata = json.loads(metadata_input)
            except Exception as e:
                print(f"Failed to parse metadata: {e}")

        thumb_path = str(request_dir / "thumbnail.webp")
        convert_to_webp_with_metadata(thumbnail, metadata, quality=90, output_path=thumb_path)

        return thumb_path, str(nsfw_level), quality_score, thumbhash
    except Exception as err:
        raise gr.Error(f"Generation failed: {err}") from err
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Media Scorer", elem_classes=["text-center"])
    gr.Markdown(
        "Score the quality (based on **human preferences**) and NSFW level of images and videos. Built with [turbogen](https://github.com/sitatec/turbogen.git)",
    )

    with gr.Tabs():
        with gr.Tab("Image Scoring"):
            with gr.Row():
                with gr.Column(scale=2):
                    img_input_mode = gr.Radio(["File Upload", "URL"], label="Input Mode", value="File Upload")
                    img_file_upload = gr.File(label="Upload Image", file_types=["image"], type="filepath", visible=True)
                    img_url_input = gr.Textbox(label="Image URL", placeholder="https://...", visible=False)
                    img_metadata_input = gr.Textbox(label="Optional Metadata (JSON)", lines=3)
                    img_submit_btn = gr.Button("Score Image", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        img_thumb_out = gr.Image(label="Thumbnail", type="filepath", scale=1)
                        with gr.Column(scale=1):
                            img_nsfw_out = gr.Textbox(label="NSFW Level")
                            img_quality_out = gr.Number(label="Quality Score")
                            img_hash_out = gr.Textbox(label="ThumbHash")

            def toggle_img_input(mode):
                return [
                    gr.update(visible=(mode == "File Upload")),
                    gr.update(visible=(mode == "URL")),
                ]

            img_input_mode.change(toggle_img_input, inputs=[img_input_mode], outputs=[img_file_upload, img_url_input])
            img_submit_btn.click(
                process_image,
                inputs=[img_input_mode, img_file_upload, img_url_input, img_metadata_input],
                outputs=[img_thumb_out, img_nsfw_out, img_quality_out, img_hash_out],
            )

        with gr.Tab("Video Scoring"):
            with gr.Row():
                with gr.Column(scale=2):
                    vid_input_mode = gr.Radio(["File Upload", "URL"], label="Input Mode", value="File Upload")
                    vid_file_upload = gr.File(label="Upload Video", file_types=["video"], type="filepath", visible=True)
                    vid_url_input = gr.Textbox(label="Video URL", placeholder="https://...", visible=False)
                    vid_metadata_input = gr.Textbox(label="Optional Metadata (JSON)", lines=3)
                    vid_submit_btn = gr.Button("Score Video", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        vid_thumb_out = gr.Image(label="Thumbnail", type="filepath", scale=1)
                        with gr.Column(scale=1):
                            vid_nsfw_out = gr.Textbox(label="NSFW Level")
                            vid_quality_out = gr.Number(label="Quality Score")
                            vid_hash_out = gr.Textbox(label="ThumbHash")

            def toggle_vid_input(mode):
                return [
                    gr.update(visible=(mode == "File Upload")),
                    gr.update(visible=(mode == "URL")),
                ]

            vid_input_mode.change(toggle_vid_input, inputs=[vid_input_mode], outputs=[vid_file_upload, vid_url_input])
            vid_submit_btn.click(
                process_video,
                inputs=[vid_input_mode, vid_file_upload, vid_url_input, vid_metadata_input],
                outputs=[vid_thumb_out, vid_nsfw_out, vid_quality_out, vid_hash_out],
            )

if __name__ == "__main__":
    app.launch(allowed_paths=[str(REQUESTS_DIR)])
