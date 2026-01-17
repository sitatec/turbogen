import json
from typing import Union, Literal, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dim: int = 1

    use_special_tokens: bool = False
    special_tokens: list[str] = field(
        default_factory=lambda: ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
    )

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: Literal[
        "bt", "reg", "btt", "margin", "constant_margin", "scaled", "regular"
    ] = "regular"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataConfig:
    meta_data_test: str | None = None
    max_frame_pixels: int = 240 * 320
    num_frames: float | None = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: Union[str, list[str]] = "VQ"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return (
        config_dict["model_config"],
        config_dict["data_config"],
        config_dict["inference_config"],
    )


VIDEOSCORE_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the frames of a given video and see the text prompt for generating the video,
then give scores based on its {dimension_name}, i.e., {dimension_description}.
Output a float number from 1.0 to 5.0 for this dimension,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 5.0 means Perfect/Real (the video is like a real video).
The text prompt used for generation is "{text_prompt}".
"""

DIMENSION_DESCRIPTIONS = {
    "VQ": [
        "visual quality",
        "the quality of the video in terms of clearness, resolution, brightness, and color",
    ],
    "TA": [
        "text-to-video alignment",
        "the alignment between the text prompt and the video content and motion",
    ],
    "MQ": [
        "motion quality",
        "the quality of the motion in terms of consistency, smoothness, and completeness",
    ],
    "Overall": [
        "Overall Performance",
        "the overall performance of the video in terms of visual quality, text-to-video alignment, and motion quality",
    ],
}

SIMPLE_PROMPT = """
Please evaluate the {dimension_name} of a generated video. Consider {dimension_description}.
The text prompt used for generation is "{text_prompt}".
"""

DETAILED_PROMPT_WITH_SPECIAL_TOKEN = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**  
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**  
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**  
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.  
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""

DETAILED_PROMPT = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**  
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Motion Quality:**  
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.


**Text Alignment:**  
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.  
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Visual Quality, Motion Quality, and Text Alignment.
"""

SIMPLE_PROMPT_NO_PROMPT = """
Please evaluate the {dimension_name} of a generated video. Consider {dimension_description}.
"""


def build_prompt(prompt, dimension, template_type):
    if isinstance(dimension, list) and len(dimension) > 1:
        dimension_name = ", ".join([DIMENSION_DESCRIPTIONS[d][0] for d in dimension])
        dimension_name = f"overall performance({dimension_name})"
        dimension_description = "the overall performance of the video"
    else:
        if isinstance(dimension, list):
            dimension = dimension[0]
        dimension_name = DIMENSION_DESCRIPTIONS[dimension][0]
        dimension_description = DIMENSION_DESCRIPTIONS[dimension][1]

    if template_type == "none":
        return prompt
    elif template_type == "simple":
        return SIMPLE_PROMPT.format(
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            text_prompt=prompt,
        )
    elif template_type == "video_score":
        return VIDEOSCORE_QUERY_PROMPT.format(
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            text_prompt=prompt,
        )
    elif template_type == "detailed_special":
        return DETAILED_PROMPT_WITH_SPECIAL_TOKEN.format(text_prompt=prompt)
    elif template_type == "detailed":
        return DETAILED_PROMPT.format(text_prompt=prompt)
    else:
        raise ValueError("Invalid template type")
