from core.services.media_scoring.image_scorer import ImageScorer
from core.services.nsfw_detector import NsfwDetector
import asyncio
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

import spaces
import aiohttp
import gradio as gr

sys.path.insert(0, ".")

from model_downloads import download_qwen_models, download_image_scorer
from core.generation_pipeline import GenerationPipeline, ProcessedOutput
from models.lightx2v_models import QwenImageLite, QwenImageEditLite


async def download_image(session: aiohttp.ClientSession, url: str, output_path: str):
    """Download a single image from URL asynchronously."""
    async with session.get(url.strip()) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(await response.read())
        return output_path


async def download_images_async(urls: list[str], request_dir: str) -> list[str]:
    """Download multiple images asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, url in enumerate(urls):
            ext = Path(url.split("?")[0]).suffix
            output_path = f"{request_dir}/input_{i}{ext}"
            tasks.append(download_image(session, url, output_path))
        return await asyncio.gather(*tasks)


def create_model_interface(
    pipeline: GenerationPipeline,
    model_id: str,
    default_negative_prompt: str | None,
    aspect_ratios: dict[str, dict[str, tuple[int, int]]],
    max_input_images: int = 0,
    postprocessing_supported: bool = False,
):
    """
    Create a reusable Gradio interface for image generation models.

    Args:
        pipeline: The generation pipeline instance
        model_id: Model identifier for the pipeline
        default_negative_prompt: Default negative prompt from model
        aspect_ratios: Dictionary of available aspect ratios and resolutions
        max_input_images: Maximum number of input images (0 for text-to-image models)
        postprocessing_supported: Whether postprocessing (NSFW, quality scoring,...) is supported
    """

    aspect_ratio_choices = list(aspect_ratios.keys())
    default_aspect_ratio = aspect_ratio_choices[0] if aspect_ratio_choices else "1:1"
    default_resolution = list(aspect_ratios[default_aspect_ratio].keys())[0]
    with gr.Row():
        with gr.Column():
            input_mode = None
            input_images_upload = None
            input_images_url = None

            if max_input_images > 0:
                with gr.Group():
                    input_images_upload = gr.File(
                        label=f"Input Images (max {max_input_images})",
                        file_count="multiple" if max_input_images > 1 else "single",
                        file_types=["image"],
                        type="filepath",
                        visible=True,
                    )

                    input_images_url = gr.Textbox(
                        label=f"Image URLs (comma-separated, max {max_input_images})",
                        placeholder="https://gen.ai/img1.jpg, https://gen.ai/img2.jpg",
                        lines=2,
                        visible=False,
                    )
                    input_mode = gr.Radio(
                        label="Input Mode",
                        choices=["File Upload", "Image URL"],
                        value="File Upload",
                    )

            with gr.Group():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    max_lines=5,
                )

                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=aspect_ratio_choices,
                        value=default_aspect_ratio,
                    )

                    resolution = gr.Dropdown(
                        label="Resolution",
                        choices=list(aspect_ratios[default_aspect_ratio].keys()),
                        value=default_resolution,
                    )

            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0,
                    info="Use -1 for random seed",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What to avoid in the image...",
                    lines=2,
                    value=default_negative_prompt,
                )

                postprocess_checkbox = None
                if postprocessing_supported:
                    postprocess_checkbox = gr.Checkbox(
                        label="Enable Postprocessing (NSFW Detection & Quality Scoring)",
                        value=False,
                    )

            generate_btn = gr.Button(
                "✨ Generate",
                variant="primary",
                size="lg",
            )

        with gr.Column():
            output_image = gr.Image(
                label="Generated Image",
                type="filepath",
            )

            # Postprocessing outputs (only visible when postprocessing is enabled)
            with gr.Row(visible=False) as postprocess_row:
                thumbnail_image = gr.Image(
                    label="Thumbnail",
                    type="filepath",
                    scale=1,
                )
                with gr.Column(scale=1):
                    nsfw_level = gr.Textbox(
                        label="NSFW Level",
                        interactive=False,
                    )
                    quality_score = gr.Number(
                        label="Quality Score",
                        interactive=False,
                    )
                    thumbhash = gr.Textbox(
                        label="ThumbHash",
                        interactive=False,
                    )

        # Toggle visibility of input image components based on mode
        if input_mode is not None:

            def toggle_input_mode(mode):
                return [
                    gr.update(visible=(mode == "File Upload")),
                    gr.update(visible=(mode == "Image URL")),
                ]

            input_mode.change(
                fn=toggle_input_mode,
                inputs=[input_mode],
                outputs=[input_images_upload, input_images_url],
            )

        # Update resolution choices when aspect ratio changes
        def update_resolution_choices(aspect_ratio_value, current_resolution):
            new_resolutions = list(aspect_ratios[aspect_ratio_value].keys())

            resolution = (
                current_resolution
                if current_resolution in new_resolutions
                else new_resolutions[0]
            )

            return gr.Dropdown(choices=new_resolutions, value=resolution)

        aspect_ratio.change(
            fn=update_resolution_choices,
            inputs=[aspect_ratio, resolution],
            outputs=[resolution],
        )

        async def prepare_inputs(
            prompt_value,
            aspect_ratio_value,
            resolution_value,
            negative_prompt_value,
            seed_value,
            postprocess_value,
            input_mode_value=None,
            input_images_upload_value=None,
            input_images_url_value=None,
        ):
            """Validate inputs and download images if needed."""
            if not prompt_value:
                raise gr.Error("Please provide a prompt!")

            request_dir = mkdtemp()

            image_paths = []
            if max_input_images > 0:
                if input_mode_value == "URL":
                    if not input_images_url_value:
                        raise gr.Error("Please provide image URLs!")

                    urls = [
                        url.strip()
                        for url in input_images_url_value.split(",")
                        if url.strip()
                    ]
                    if not urls:
                        raise gr.Error("Please provide valid image URLs!")

                    urls = urls[:max_input_images]

                    try:
                        image_paths = await download_images_async(urls, request_dir)
                    except Exception as e:
                        if Path(request_dir).exists():
                            shutil.rmtree(request_dir)
                        raise gr.Error(f"Failed to download images: {str(e)}")
                else:  # Upload mode
                    if not input_images_upload_value:
                        raise gr.Error("Please upload at least one image!")

                    if isinstance(input_images_upload_value, list):
                        image_paths = input_images_upload_value[:max_input_images]
                    else:
                        image_paths = [input_images_upload_value]

            return {
                "request_dir": request_dir,
                "image_paths": image_paths,
                "prompt": prompt_value,
                "aspect_ratio": aspect_ratio_value,
                "resolution": resolution_value,
                "negative_prompt": negative_prompt_value
                if negative_prompt_value
                else None,
                "seed": int(seed_value),
                "postprocess": postprocess_value if postprocessing_supported else False,
            }

        @spaces.GPU
        def generate_on_gpu(prepared_inputs):
            return pipeline.generate(
                model_id=model_id,
                prompt=prepared_inputs["prompt"],
                aspect_ratio=prepared_inputs["aspect_ratio"],
                resolution=prepared_inputs["resolution"],
                image_paths=prepared_inputs["image_paths"],
                seed=prepared_inputs["seed"],
                negative_prompt=prepared_inputs["negative_prompt"],
                postprocess=prepared_inputs["postprocess"],
                output_dir_path=prepared_inputs["request_dir"],
            )

        async def generate(
            prompt_value,
            aspect_ratio_value,
            resolution_value,
            negative_prompt_value,
            seed_value,
            postprocess_value=False,
            input_mode_value=None,
            input_images_upload_value=None,
            input_images_url_value=None,
        ):
            """Main generation function that coordinates preprocessing and GPU execution."""
            request_dir = None
            try:
                prepared_inputs = await prepare_inputs(
                    prompt_value,
                    aspect_ratio_value,
                    resolution_value,
                    negative_prompt_value,
                    seed_value,
                    postprocess_value,
                    input_mode_value,
                    input_images_upload_value,
                    input_images_url_value,
                )

                request_dir = prepared_inputs["request_dir"]

                result = generate_on_gpu(prepared_inputs)

                if isinstance(result, ProcessedOutput):
                    return (
                        result.generated_media_path,
                        gr.update(visible=True),
                        result.thumbnail_path,
                        result.nsfwLevel.value,
                        result.quality_score,
                        result.thumbhash,
                    )
                else:
                    # No postprocessing - just return the image path
                    return (
                        result,
                        gr.update(visible=False),
                        None,
                        None,
                        None,
                        None,
                    )
            except Exception as e:
                raise gr.Error(f"Generation failed: {str(e)}")
            finally:
                # Clean up temporary directory
                if request_dir and Path(request_dir).exists():
                    try:
                        shutil.rmtree(request_dir)
                    except Exception:
                        pass

        inputs_list: list = [
            prompt,
            aspect_ratio,
            resolution,
            negative_prompt,
            seed,
        ]

        if postprocessing_supported:
            inputs_list.append(postprocess_checkbox)

        if max_input_images > 0:
            inputs_list.extend([input_mode, input_images_upload, input_images_url])

        generate_btn.click(
            fn=generate,
            inputs=inputs_list,
            outputs=[
                output_image,
                postprocess_row,
                thumbnail_image,
                nsfw_level,
                quality_score,
                thumbhash,
            ],
        )

    return {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "postprocess_checkbox": postprocess_checkbox,
        "input_mode": input_mode,
        "input_images_upload": input_images_upload,
        "input_images_url": input_images_url,
        "output_image": output_image,
    }


def create_app(
    pipeline: GenerationPipeline,
    title: gr.Component,
    postprocessing_supported: bool = False,
):
    """Create the main Gradio application with tabs for different models."""

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        title

        with gr.Tabs():
            for model in pipeline.models:
                max_input_images = getattr(model, "max_input_images", 0)

                with gr.Tab(model.model_name):
                    create_model_interface(
                        pipeline=pipeline,
                        model_id=model.model_id,
                        aspect_ratios=model.supported_aspect_ratios,
                        max_input_images=max_input_images,
                        default_negative_prompt=model.default_negative_prompt,
                        postprocessing_supported=postprocessing_supported,
                    )

    return app


if __name__ == "__main__":
    qwen_image_edit_path, qwen_image_path = download_qwen_models()
    image_scorer_path = download_image_scorer()

    qwen_image = QwenImageLite(qwen_image_path)
    qwen_image_edit = QwenImageEditLite(qwen_image_edit_path)

    pipeline = GenerationPipeline(
        models=[qwen_image, qwen_image_edit],
        nsfw_detector=NsfwDetector(),
        image_scorer=ImageScorer(image_scorer_path),
        video_scorer=None,
    )

    app = create_app(
        pipeline,
        postprocessing_supported=False,
        title=gr.Markdown(
            """
            # 🎨 Qwen Image Generation and Editing
            Create stunning images with the latest Qwen models
            """,
            elem_classes=["text-center"],
        ),
    )
    app.launch()
