from __future__ import annotations

import os
import uuid
import shutil
import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Any


import spaces
import aiohttp
import gradio as gr
import numpy as np

if TYPE_CHECKING:
    from core.models.base_model import BaseModel
    from core.generation_pipeline import GenerationPipeline, ProcessedOutput

pipe: GenerationPipeline | None = None


def get_gen_duration(inputs: dict):
    assert pipe is not None
    num_outputs = inputs.get("num_outputs", 1)
    duration = 50
    initialization_time = 15  # Estimated Zero GPU initialization time
    if inputs.get("enhance_prompt"):
        initialization_time += 15

    model = next(
        (model for model in pipe.models if model.model_id == inputs["model_id"]),
    )

    model_name = model.model_name.lower()
    if model_name.startswith("qwen"):
        if model.generation_type.value.lower() == "t2i":
            duration = 4
        else:
            duration = 6
    elif model_name.startswith("wan"):
        if inputs["resolution"] == "480p":
            duration = 20
        else:
            duration = 50
    elif model_name.startswith("z-image-turbo"):
        if inputs["resolution"] == "1k":
            duration = 5
        else:
            duration = 7

    return duration * num_outputs + initialization_time


@spaces.GPU(duration=get_gen_duration)
def generate_on_gpu(prepared_inputs: dict):
    assert pipe is not None

    num_outputs = prepared_inputs.get("num_outputs", 1)

    seed = prepared_inputs.get("seed", -1)
    if seed == -1:
        seed = random.randint(1, np.iinfo(np.int32).max)

    for i in range(num_outputs):
        output_dir = prepared_inputs["request_dir"] / f"output_{i}"
        output_dir.mkdir(parents=True, exist_ok=True)

        yield pipe.generate(
            model_id=prepared_inputs["model_id"],
            prompt=prepared_inputs["prompt"],
            aspect_ratio=prepared_inputs["aspect_ratio"],
            resolution=prepared_inputs["resolution"],
            image_paths=prepared_inputs["image_paths"],
            last_frame_path=prepared_inputs["last_frame_path"],
            seed=seed,
            negative_prompt=prepared_inputs["negative_prompt"],
            postprocess=prepared_inputs["postprocess"],
            enhance_prompt=prepared_inputs["enhance_prompt"],
            output_dir_path=output_dir,
            metadata=prepared_inputs.get("metadata"),
        )

        seed += 1


async def download_file(session: aiohttp.ClientSession, url: str, output_path: str):
    """Download a single image from URL asynchronously."""
    async with session.get(url.strip()) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(await response.read())
        return output_path


async def download_files(urls: list[str], request_dir: str | Path) -> list[str]:
    """Download multiple images asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, url in enumerate(urls):
            ext = Path(url.split("?")[0]).suffix
            output_path = f"{request_dir}/input_{i}{ext}"
            tasks.append(download_file(session, url, output_path))
        return await asyncio.gather(*tasks)


async def call_callback(callback, *args, **kwargs):
    if asyncio.iscoroutinefunction(callback):
        return await callback(*args, **kwargs)
    else:
        return callback(*args, **kwargs)


def create_model_interface(
    pipeline: GenerationPipeline,
    model_id: str,
    model: BaseModel,
    default_negative_prompt: str | None,
    aspect_ratios: dict[str, dict[str, tuple[int, int]]],
    max_input_images: int = 0,
    postprocessing_supported: bool = False,
    prompt_enhancing_supported: bool = False,
    pre_gen_hook: Callable[[dict], dict | None] | None = None,
    post_gen_hook: Callable[[list[ProcessedOutput | str], gr.Request, Any], None]
    | None = None,
    inference_dir: str = "/tmp/inference_requests",
):
    """
    Create a reusable Gradio interface for image and video generation models.

    Args:
        pipeline: The generation pipeline instance
        model_id: Model identifier for the pipeline
        default_negative_prompt: Default negative prompt from model
        aspect_ratios: Dictionary of available aspect ratios and resolutions
        max_input_images: Maximum number of input images
        postprocessing_supported: Whether postprocessing (NSFW, quality scoring,...) is supported
        prompt_enhancing_supported: Whether or not user can choose to enhance the input prompt.
        pre_gen_hook: Called before generation starts (can be for used to prepare metadata for e.g.), this was added for internal use, it is currently not used in this repo, but feel free to use it.
        post_gen_hook: Called after generation ends, this was added for internal use, it is currently not used in this repo, but feel free to use it.
    """

    is_video = model.generation_type.is_video
    media_type_label = "Video" if is_video else "Image"

    aspect_ratio_choices = list(aspect_ratios.keys())
    default_aspect_ratio = aspect_ratio_choices[0] if aspect_ratio_choices else "1:1"
    default_resolution = list(aspect_ratios[default_aspect_ratio].keys())[0]

    with gr.Row():
        with gr.Column():
            input_mode = None
            input_image_upload = None
            input_image_url = None
            last_frame_upload = None
            last_frame_url = None

            if max_input_images > 0:
                with gr.Group():
                    if is_video:
                        input_image_upload = gr.Image(
                            label="First Frame",
                            type="filepath",
                            visible=True,
                        )
                        input_image_url = gr.Textbox(
                            label="First Frame URL",
                            placeholder="https://gen.ai/first_frame.jpg",
                            visible=False,
                        )
                        with gr.Accordion("Last Frame (Optional)", open=False):
                            last_frame_upload = gr.Image(
                                label="Last Frame",
                                type="filepath",
                                visible=True,
                            )
                            last_frame_url = gr.Textbox(
                                label="Last Frame URL",
                                placeholder="https://gen.ai/last_frame.jpg",
                                visible=False,
                            )
                    else:
                        # For image, we have multi-input images
                        input_image_upload = gr.File(
                            label=f"Input Images (max {max_input_images})",
                            file_count="multiple" if max_input_images > 1 else "single",
                            file_types=["image"],
                            type="filepath",
                            visible=True,
                        )
                        input_image_url = gr.Textbox(
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
                    placeholder=f"Describe the {media_type_label.lower()} you want to generate...",
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
                    placeholder="What to avoid in the generation...",
                    lines=2,
                    value=default_negative_prompt,
                )

                num_outputs = gr.Slider(
                    label="Number of Outputs",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                )

                prompt_enhancer_checkbox = None
                if prompt_enhancing_supported:
                    prompt_enhancer_checkbox = gr.Checkbox(
                        label="Enhance Prompt",
                        value=True,
                    )

                postprocess_checkbox = None
                if postprocessing_supported:
                    postprocess_checkbox = gr.Checkbox(
                        label="Enable Postprocessing (NSFW Detection & Quality Scoring)",
                        value=False,
                    )

            generate_btn = gr.Button(
                f"✨ Generate {media_type_label}",
                variant="primary",
                size="lg",
            )

        with gr.Column():
            output_media = gr.Gallery(
                label=f"Generated {media_type_label}s",
                columns=2,
                height="auto",
                object_fit="contain",
            )

            # Postprocessing outputs (only visible when postprocessing is enabled)
            with gr.Row(visible=False) as postprocess_row:
                thumbnail = gr.Image(
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

        # Toggle visibility of input components based on mode
        if input_mode is not None:

            def toggle_input_mode(mode):
                updates = [
                    gr.update(visible=(mode == "File Upload")),
                    gr.update(visible=(mode == "Image URL")),
                ]
                if is_video:  # Last frame
                    updates.extend(
                        [
                            gr.update(visible=(mode == "File Upload")),
                            gr.update(visible=(mode == "Image URL")),
                        ]
                    )
                return updates

            outputs = [input_image_upload, input_image_url]
            if is_video:
                outputs.extend([last_frame_upload, last_frame_url])

            input_mode.change(
                fn=toggle_input_mode,
                inputs=[input_mode],
                outputs=outputs,  # pyrefly: ignore
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
            prompt_enhancer_value,
            num_outputs_value,
            request,
            model,
            input_mode_value=None,
            input_image_upload_value=None,
            input_image_url_value=None,
            last_frame_upload_value=None,
            last_frame_url_value=None,
        ):
            """Validate inputs and download media if needed."""
            if not prompt_value:
                raise gr.Error("Please provide a prompt!")

            request_dir = Path(f"{inference_dir}/{uuid.uuid4()}")
            request_dir.mkdir(parents=True, exist_ok=True)
            image_paths = []
            last_frame_path = None

            # We preserve the original urls for metadata.
            image_urls = None
            last_frame_url = None

            if max_input_images > 0:
                if input_mode_value == "Image URL":
                    if not input_image_url_value:
                        raise gr.Error("Please provide media URLs!")

                    urls = [
                        url.strip()
                        for url in input_image_url_value.split(",")
                        if url.strip()
                    ]
                    if not urls:
                        raise gr.Error("Please provide valid media URLs!")
                    image_urls = urls
                    if is_video:
                        if last_frame_url_value:
                            last_frame_url = last_frame_url_value
                            urls.append(last_frame_url_value)
                        try:
                            downloaded_paths = await download_files(urls, request_dir)
                            image_paths = [downloaded_paths[0]]
                            if len(downloaded_paths) > 1:
                                last_frame_path = downloaded_paths[1]
                        except Exception as e:
                            if Path(request_dir).exists():
                                shutil.rmtree(request_dir)
                            raise gr.Error(f"Failed to download media: {str(e)}")
                    else:
                        # For image, all URLs are input images
                        urls = urls[:max_input_images]
                        try:
                            image_paths = await download_files(urls, request_dir)
                        except Exception as e:
                            if request_dir.exists():
                                shutil.rmtree(request_dir)
                            raise gr.Error(f"Failed to download images: {str(e)}")
                else:  # File Upload mode
                    if not input_image_upload_value:
                        raise gr.Error("Please upload at least one image!")

                    if is_video:
                        image_paths = [input_image_upload_value]
                        last_frame_path = last_frame_upload_value
                    else:
                        if isinstance(input_image_upload_value, list):
                            image_paths = input_image_upload_value[:max_input_images]
                        else:
                            image_paths = [input_image_upload_value]

            prepared_inputs = {
                "model_id": model_id,
                "request_dir": request_dir,
                "image_paths": image_paths,
                "last_frame_path": last_frame_path,
                "prompt": prompt_value,
                "aspect_ratio": aspect_ratio_value,
                "resolution": resolution_value,
                "negative_prompt": negative_prompt_value
                if negative_prompt_value
                else None,
                "seed": int(seed_value),
                "num_outputs": int(num_outputs_value),
                "postprocess": postprocess_value if postprocessing_supported else False,
                "enhance_prompt": prompt_enhancer_value
                if prompt_enhancing_supported
                else False,
            }

            if pre_gen_hook:
                metadata = await call_callback(
                    pre_gen_hook,
                    {
                        "model": model,
                        "request": request,
                        "image_urls": image_urls,
                        "last_frame_url": last_frame_url,
                        **prepared_inputs,
                    },
                )
                if metadata:
                    prepared_inputs["metadata"] = metadata

            return prepared_inputs

        async def generate(
            prompt_value,
            aspect_ratio_value,
            resolution_value,
            negative_prompt_value,
            seed_value,
            num_outputs_value,
            prompt_enhancer_value=False,
            postprocess_value=False,
            input_mode_value=None,
            input_image_upload_value=None,
            input_image_url_value=None,
            last_frame_upload_value=None,
            last_frame_url_value=None,
            request: gr.Request = gr.Request(),
            progress=gr.Progress(track_tqdm=True),
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
                    prompt_enhancer_value,
                    num_outputs_value,
                    request,
                    model,
                    input_mode_value,
                    input_image_upload_value,
                    input_image_url_value,
                    last_frame_upload_value,
                    last_frame_url_value,
                )

                request_dir = prepared_inputs["request_dir"]

                all_outputs = []
                for output in generate_on_gpu(prepared_inputs):
                    all_outputs.append(output)

                    if isinstance(output, str):
                        yield (
                            all_outputs,
                            gr.update(visible=False),
                            None,
                            None,
                            None,
                            None,
                        )
                    else:
                        yield (
                            [out.generated_media_path for out in all_outputs],
                            gr.update(visible=True),
                            output.thumbnail_path,
                            output.nsfwLevel.value,
                            output.quality_score,
                            output.thumbhash,
                        )

                if post_gen_hook:
                    await call_callback(post_gen_hook, all_outputs, request, None)
            except Exception as e:
                raise gr.Error(f"Generation failed: {str(e)}")
            finally:
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
            num_outputs,
        ]

        if prompt_enhancing_supported:
            inputs_list.append(prompt_enhancer_checkbox)

        if postprocessing_supported:
            inputs_list.append(postprocess_checkbox)

        if max_input_images > 0:
            inputs_list.extend(
                [
                    input_mode,
                    input_image_upload,
                    input_image_url,
                ]
            )
            if is_video:
                inputs_list.extend(
                    [
                        last_frame_upload,
                        last_frame_url,
                    ]
                )

        generate_btn.click(
            fn=generate,
            show_progress_on=output_media,
            inputs=inputs_list,
            outputs=[
                output_media,
                postprocess_row,
                thumbnail,
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
        "num_outputs": num_outputs,
        "postprocess_checkbox": postprocess_checkbox,
        "prompt_enhancer_checkbox": prompt_enhancer_checkbox,
        "input_mode": input_mode,
        "input_image_upload": input_image_upload,
        "input_image_url": input_image_url,
        "output_media": output_media,
    }


def create_gradio_app(
    pipeline: GenerationPipeline,
    title: str,
    postprocessing_supported: bool = False,
    prompt_enhancing_supported: bool = False,
    pre_gen_hook: Callable[[dict], dict | None] | None = None,
    post_gen_hook: Callable[[list[ProcessedOutput | str], gr.Request, Any], None]
    | None = None,
    inference_dir: str = "/tmp/inference_requests",
):
    """Create the main Gradio application with tabs for different models."""
    global pipe
    pipe = pipeline

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown(
            title,
            elem_classes=["text-center"],
        )

        with gr.Tabs():
            for model in pipeline.models:
                max_input_images = getattr(model, "max_input_images", 0)

                if max_input_images == 0 and model.generation_type.value.lower() in [
                    "i2i",
                    "i2v",
                ]:
                    max_input_images = 1

                with gr.Tab(model.model_name):
                    create_model_interface(
                        pipeline=pipeline,
                        model_id=model.model_id,
                        aspect_ratios=model.supported_aspect_ratios,
                        max_input_images=max_input_images,
                        default_negative_prompt=model.default_negative_prompt,
                        postprocessing_supported=postprocessing_supported,
                        prompt_enhancing_supported=prompt_enhancing_supported,
                        model=model,
                        pre_gen_hook=pre_gen_hook,
                        post_gen_hook=post_gen_hook,
                        inference_dir=inference_dir,
                    )

    os.environ["GRADIO_ALLOWED_PATHS"] = inference_dir

    return app


__all__ = [create_gradio_app]
