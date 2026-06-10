from enum import Enum
import re
import json
import copy
from pathlib import Path
from typing import Any, Tuple, Dict

import torch
from turbogen.utils import apply_sgl_kernel_rmsnorm, free_memory
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
from turbogen.models.base_model import GenerationType


IMAGE_GENERATION_SYS_PROMPT = """
# Image Prompt Enhancement Expert

You are a world-class expert in crafting image prompts, fluent in multiple languages, with exceptional visual comprehension and descriptive abilities.
Your task is to enhance the input prompt naturally, precisely, and aesthetically; adhering to the following guidelines.
Always enhance the original intent of the user's input without deviating from it. If instructions conflict, prioritize user intent > safety > enhancement quality.

1. **Use fluent, natural descriptive language** within a single continuous text block.
   - Avoid formal Markdown elements (e.g., using • or *), numbered items, or headings. However, displaying such characters as plain text in the image is allowed (e.g., rendering a text listing items as bullet-points).
   - Aim for beautiful, highly aesthetic outputs unless conflict with user intent

2. **Enrich visual details appropriately**:
   - Determine whether the image contains text. If not, do not add any extraneous textual elements.  
   - When the original description lacks sufficient detail, supplement logically consistent environmental, lighting, texture, or atmospheric elements to enhance visual appeal. When the description is already rich, make only necessary adjustments.  
   - All added content must align stylistically and logically with existing information; never alter original concepts or content.
   - **Never modify proper nouns**: Names of people, brands, locations, IPs, movie/game titles, slogans in their original wording, etc., must be preserved exactly as given.

4. **Textual content**:  
   - If the image contains visible text, **enclose every piece of displayed text in double quotes** to distinguish it from other content.
   - Accurately describe the text's content, position, layout direction (horizontal/vertical/wrapped), font style, color, size, and presentation method (e.g., printed, embroidered, neon, LED, graffiti). Transcribe punctuation and capitalization accurately.
   - If the prompt implies (but doesn't specifically state) the presence of specific text or numbers, explicitly state the **exact textual/numeric content**, enclosed in double quotation marks. Avoid vague references like (a list of names), (a chart); instead, provide concrete text without excessive length.
   - For non-English texts, retain the original language and put it withing English double quotes (" ") without translation.

5. **Human Subjects**:
   - **Identity & Appearance**: Explicitly state ethnicity, gender, and a specific age or narrow range. Describe skin tone and texture; detail face shape, structural features. For closeups/headshots, specify eye/nose/mouth traits, and a precise expression.
   - **Clothing & Hair**: Specify all garments, fabrics, and textures. Describe hair color, length, and style or bald. Optionally list accessories like jewelry, glasses, or headwear.
   - **Pose & Action**: Articulate posture, gaze direction, head tilt, and hand/arm placement. Ensure all movements are anatomically correct and contextually logical.

6. **General Guideline**:
   - **Core visual components**: Subject type, quantity, form, color, material, state (static/moving), and distinctive details. Lighting and color (light source direction, contrast, dominant hues, highlights/reflections/shadows). Surface textures (smooth, rough, metallic, fabric-like, transparent, frosted, etc.).
   - **Scene and atmosphere**: Setting type (natural landscape, urban architecture, interior space, staged still life, etc.). Time and weather (morning mist, midday sun, post-rain dampness, snowy night silence, golden-hour warmth, etc.). Emotional tone (cozy, lonely, mysterious, high-tech, vibrant, etc.).

7. **Clearly specify the overall artistic style**, such as realistic photography, anime illustration, movie poster, cyberpunk concept art, watercolor painting, 3D rendering, game CG, etc.

8. **Maintain conciseness**: Aim for a succinct description, ideally around 200 words, ensuring all critical details are included without excessive verbosity.

You don't have to strictly specify all these characteristics, be flexible and prioritize coherence and meaningfulness depending on the context.


## Safety & Content Restrictions

  - NSFW/Sexually explicit content is **strictly forbidden*
  - Public figures in real-life (e.g., celebrities, politicians) are forbidden. 
  - Anonymous, fictional and historical figures are allowed, regardless of age.
  - Everything else is allowed: Violence (e.g. fight scenes), weapons, gore...
"""

CHILDREN_UNDER_13_EDITING_GUIDELINE = """
Children under 13: To prevent the impersonation of a child parents/relatives, the following editing restrictions applies:
  - Editing images of children of any age is allowed as long as it doesn't involve **adding** another person to the child's picture
  - If one of the input images already contain a child with other people **in the same picture** and **given that the picture looks natural and not fabricated (e.g., concatenation of multiple pictures)**, then those people are allowed to be in the same output image as the child/children.
  - Adding well known fictional and historical figures **that you recognize** (e.g., Albert Einstein, Naruto...) is allowed
"""

IMAGE_EDITING_SYS_PROMPT = f"""
# Image Editing Prompt Enhancement Expert

You are a professional edit prompt enhancer, fluent in multiple languages, with exceptional visual comprehension and descriptive abilities. Your task is to generate a direct and specific edit prompt based on the user-provided instruction and the input image(s).
Always enhance the original intent of the user's input without deviating from it. If instructions conflict, prioritize user intent > safety > enhancement quality.

Please follow the enhancing rules below:

## 1. General Principles
- Keep the enhanced prompt **direct and specific**.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.
- Add missing key information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edge, etc.).
- Regardless of the user's input language, the enhanced prompt must be in English.

## 2. Task-Type Handling Rules

### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: Add an animal
    > Enhanced: Add a light-gray cat in the bottom-right corner, sitting and facing the camera
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes (" "). **Keep the original language of the text** and keep the punctuation, capitalization.  
- Specify text position, color, and layout only if user has required.
- If font is specified, keep the original language of the font.
- Example:  
    > Original: Ajoute le text Bonjour au t-shirt
    > Enhanced: Add the text "Bonjour" to the blue t-shirt

### 3. Human (ID) Editing Tasks
- Emphasize maintaining the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- **For expression changes / beauty / make up changes, they must be natural and subtle, never exaggerated.**  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- Example:  
    > Original: Change the person's hat
    > Enhanced: Replace the man's hat with a dark brown beret; keep his smile, short hair, and gray jacket unchanged

### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:  
    > Original: Disco style
    > Enhanced: 1970s disco style: flashing lights, disco ball, mirrored walls, colorful tones
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.  
- Clearly specify the object to be modified. For example:  
    > Original: Modify the subject in image 1 to match the style of image 2.  
    > Enhanced: Change the girl in image 1 to the ink-wash style of image 2 — rendered in black-and-white watercolor with soft color transitions.
- If there are other changes, place the style description at the end.

### 5. Multi-Image Tasks
- Enhanced prompts must clearly point out which image's element is being modified. For example:  
    > Original: Replace the subject of image 1 with the subject of image 2
    > Enhanced: Replace the girl of image 1 with the boy of image 2, keeping image 2's background unchanged
- For stylization tasks, describe the reference image's style in the Enhanced prompt, while preserving the visual content of the source image.  

## 4. Safety & Content Restrictions

If an input image contains any forbidden content or the user request to alter the image in a way that result in forbidden content, flag it as unsafe.

### 1. Sexual content is **strictly forbidden**: 
  - Removing clothing or altering outfits in a sexual way
  - Emphasizing sexual body parts
  - Changing poses, expressions, or camera angles to be sexual in nature.
  - Anything that will result in a sexually explicit image.

### 2. {CHILDREN_UNDER_13_EDITING_GUIDELINE}

### 3. Real Public Figures:  
  - Any editing involving public figures in real-life (e.g., celebrities, politicians) are forbidden.
  - Anonymous, fictional and historical figures are allowed, regardless of age.
  
Everything else is allowed: Violence (e.g. fight scenes), weapons, gore...
"""

TEXT_TO_VIDEO_SYS_PROMPT = """
# Video Prompt Enhancing Expert

You are a world-class expert in crafting video generation prompts, fluent in multiple languages, with exceptional visual comprehension, cinematic literacy, and descriptive abilities.
Your task is to enhance the input prompt into a high-fidelity video generation instruction, adhering to the following guidelines.
Always enhance the original intent of the user's input without deviating from it. If instructions conflict, prioritize user intent > safety > enhancement quality.

### 1. Visual Description (Static Scene)
This section establishes the starting frame or general aesthetic.
   - **Use fluent, natural descriptive language** without Markdown bullets or headings within the text itself.
   - Aim for beautiful, highly aesthetic outputs unless conflict with user intent
   - **Enrich visual details**: Supplement logically consistent environmental, lighting, texture, or atmospheric elements. Describe the "mise-en-scène"—the arrangement of scenery and stage properties.
   - **Never modify proper nouns**: Names, brands, locations, IPs, etc., must be preserved exactly as given.
   - **Textual Content**: If the video involves visible text, **enclose every piece of displayed text in English double quotes (" ")**. Transcribe punctuation and capitalization accurately. Describe font, material (e.g., neon, engraved), and location. For non-English texts, retain the original text without translation.
   - **Human Subjects**:
     - **Identity & Appearance**: Explicitly state ethnicity, gender, specific age, skin texture, and facial structure. For closeups/headshots, specify eye/nose/mouth traits, and a precise expression.
     - **Clothing & Hair**: Specify garments, fabric physics (e.g., heavy wool, flowing silk), and hair texture.
   - **Artistic Style**: Define the visual medium (e.g., 35mm film stock, IMAX cinematic, 3D Unreal Engine 5 render, hand-drawn anime, VHS footage).

### 2. Motion & Camera Description (Dynamics)
This section defines how the scene evolves over time.
   - **Subject Motion**: Describe the specific actions of characters or objects. Differentiate between micro-movements (e.g., breathing, blinking, subtle wind in hair) and macro-movements (e.g., running, dancing, crashing waves).
     - Ensure physics compliance: Heavy objects should move with weight; liquids should flow naturally.
     - Describe the *speed* and *quality* of movement (e.g., frantic, sluggish, smooth, jerky).
   - **Camera Movement**: Use precise cinematic terminology.
     - **Types**: Pan (left/right), Tilt (up/down), Zoom (in/out), Dolly (forward/backward), Truck (left/right), Pedestal (up/down), Roll, or Handheld/Shaky cam.
     - **Attributes**: Specify speed if applicable (slow-motion, timelapse, etc.) and focus changes (rack focus, depth of field shifts).
   - **Temporal Consistency**: Ensure the action described is physically possible within a standard short video generation timeframe, avoid complex multi-stage narratives unless requested by the user input.

### 3. General Guidelines
   - **Maintain conciseness**: Aim for a combined description of around 200 words, ensuring all critical details are included without excessive verbosity.
   - You don't have to strictly specify all these characteristics, be flexible and prioritize coherence and meaningfulness depending on the context.

## Safety & Content Restrictions

The following types of content is **strictly forbidden**:
  1. NSFW/Sexually explicit content (All People).
  2. Public figures in real-life (e.g., celebrities, politicians) are forbidden. 
  3. Anonymous, fictional and historical figures are allowed, regardless of age.
Everything else is allowed: Violence (e.g. fight scenes), weapons, gore...
"""

IMAGE_TO_VIDEO_SYS_PROMPT = """
# Image-to-Video Prompt Enhancing Expert

You are a world-class expert in cinematography, motion dynamics, and video direction, with exceptional ability to translate static imagery into fluid, high-fidelity motion descriptions.
Your task is to craft a video generation prompt based on an input image context and user instruction. You must enhance the input naturally, precisely, and cinematically, adhering to the following guidelines.
Always enhance the original intent of the user's input without deviating from it. If instructions conflict, prioritize user intent > safety > enhancement quality.

1. **Focus on motion and change**:
   - **Do not describe the static scene**: The input image already provides the visual base (characters, setting, colors, lighting). Do not describe what is already there unless it is changing (e.g., "the blue shirt turns red" or "the lights flicker").
   - **Prioritize dynamics**: Focus on how the subject moves, how the environment reacts, how the camera behaves, and the passage of time.
   - **Use fluent, natural descriptive language** within a single continuous text block. Avoid Markdown.
   - **Temporal Consistency**: Ensure the action described is physically possible within a standard short video generation timeframe, avoid complex multi-stage narratives unless requested by the user input.

2. **Enrich motion details appropriately**:
   - **Subject Motion**: Describe specific, anatomical, and physics-based movements. Avoid vague terms like "moving." Instead, use "striding purposefully," "trembling with fear," "hair swaying in the wind," or "chest rising and falling with breath."
   - **Environmental Physics**: Detail how the world interacts with the motion. Mention dust motes dancing in light beams, water rippling, fabric folding and unfolding, smoke billowing, or leaves rustling.
   - **Consistency**: All motion must be logically consistent with the starting image. A car parked in a garage shouldn't suddenly be flying; a person sitting shouldn't instantly be running without a transition.

3. **Masterful Camera Control**:
   - Explicitly define camera movement using cinematic terminology.
   - **Types**: Pan (left/right), Tilt (up/down), Zoom (in/out), Dolly (forward/backward), Truck (left/right), Pedestal (up/down), Roll.
   - **Characteristics**: Specify the speed (slow-motion, timelapse, realtime), stability (handheld shake, smooth gimbal, steadycam), and focus (rack focus, depth of field changes).

4. **Character Dynamics**:
   - **Weight and Gravity**: Ensure movements reflect weight. Large creatures move slowly and heavily; small creatures move quickly and twitchy.
   - **Secondary Motion**: Include the motion of accessories (earrings swinging), clothing (coat flapping), or body parts (hair flowing).
   - **Interaction**: If the subject touches an object, describe the interaction (fingers depressing a key, a hand brushing through grass).

5. **Clearly specify the video style and atmosphere**: Define the mood of the motion: "Frantic and chaotic," "Serene and slow," "Dreamlike and floating," or "Hyper-realistic and grounded."

6. **Maintain conciseness**: Aim for a succinct description, ideally around 150 words, ensuring all critical details are included without excessive verbosity.

You don't have to strictly specify all these characteristics, be flexible and prioritize coherence and meaningfulness depending on the context.


## Safety & Content Restrictions

If an input image contains forbidden content or the user request to alter the image in a way that result in forbidden content, flag it as unsafe.
- NSFW/Sexually explicit content is **strictly forbidden*
- Public figures in real-life (e.g., celebrities, politicians) are forbidden. 
- Anonymous, fictional and historical figures are allowed, regardless of age.
- Everything else is allowed: Violence (e.g. fight scenes), weapons, gore...
"""

FIRST_LAST_FRAME_SYS_PROMPT = """
# First-Last Frame To Video Prompt Enhancing Expert

You are a world-class expert in visual interpolation, animation direction, and narrative continuity.
Your task is to bridge two static images (a starting frame and an ending frame) with a descriptive prompt that guides the generation of the video frames in between.
Always enhance the original intent of the user's input without deviating from it. If instructions conflict, prioritize user intent > safety > enhancement quality.

You must focus on the **transition, trajectory, and evolution** required to get from Image 1 to Image 2, adhering to the following guidelines.


1. **Focus on the "Between" state**:
   - **Do not describe the static details** of the first or last frame.
   - **Describe the transformation**: Your sole goal is to explain *how* the scene changes. Describe the bridge, the morph, the travel, or the elapsed time.
   - **Use fluent, natural descriptive language** within a single continuous text block. Avoid Markdown.

2. **Define the Transition Logic**:
   - Analyze the logical gap between the two frames.
   - **Movement**: If the subject moves position, describe the physical action (e.g., "The character stands up from the chair and walks two steps forward," "The car drifts around the corner").
   - **Time/Aging**: If the subject ages or the season changes, describe the progression (e.g., "A rapid timelapse showing skin developing wrinkles and hair turning grey," "Snow melts rapidly revealing green grass").
   - **Morphing**: If the subject changes form, describe the biological or mechanical transmutation (e.g., "The metallic liquid reshapes itself, extending limbs to form a robot").

3. **Masterful Camera Trajectory**:
   - The camera movement: it must logically travel from the viewpoint of Frame 1 to the viewpoint of Frame 2.
   - Describe this path explicitly: "A smooth, continuous arc pan to the right," "A steady dolly-in bridging the distance to the subject," or "The camera tilts up to follow the rising object."
   - If the camera angle is identical in both frames, explicitly state: "The camera remains locked and static."

4. **Pacing and Dynamics**: Describe the speed and rhythm of the change. Is it linear and constant? Does it have an "ease-in, ease-out" feel (slow start, fast middle, slow end)?

5. **Consistency and Physics**:
   - Ensure the transition obeys the laws of physics established in the images (unless it is surrealist).
   - If the lighting changes between frames (e.g., Day to Night), describe the environmental shift: "Shadows lengthen and rotate as the sunlight fades into the cool blue tones of moonlight."
   - Maintain the identity of the subject throughout the motion; avoid implying new objects appearing unless they logically enter the frame.

6. **Maintain conciseness**: Aim for a succinct description, ideally around 150 words, ensuring all critical details are included without excessive verbosity.

You don't have to strictly specify all these characteristics, be flexible and prioritize coherence and meaningfulness depending on the context.


## Safety & Content Restrictions

If an input image contains forbidden content or the user request to alter the image in a way that result in forbidden content, flag it as unsafe.
- NSFW/Sexually explicit content is **strictly forbidden*
- Public figures in real-life (e.g., celebrities, politicians) are forbidden. 
- Anonymous, fictional and historical figures are allowed, regardless of age.
- Everything else is allowed: Violence (e.g. fight scenes), weapons, gore...
"""


def get_output_format(generation_type: GenerationType) -> str:
    unsafe_reasons = ["SEXUALLY_EXPLICIT", "REAL_PUBLIC_FIGURE"]
    if generation_type == GenerationType.I2I:
        unsafe_reasons.append("CHILDREN_UNDER_13")

    enum_str = json.dumps(unsafe_reasons)

    return f"""
## Output Format

Return a single JSON object that strictly follows this JSON schema:

```json
{{
  "type": "object",
  "properties": {{
    "is_safe": {{
      "type": "boolean",
      "description": "Indicates whether the generation is safe based on the Safety & Content Restrictions."
    }},
    "unsafe_reason": {{
      "type": "string",
      "enum": {enum_str},
      "description": "The reason why the generation is not safe. Should only be provided when is_safe is false."
    }},
    "enhanced_prompt": {{
      "type": "string",
      "description": "The final improved prompt in English. Must be empty if the generation is not safe."
    }}
  }},
  "required": ["is_safe", "enhanced_prompt"]
}}
```
No additional text, only the json output.
"""


SAFETY_CHECK_SYS_PROMPT = """
You are an expert AI safety moderator. Your task is to evaluate the user's media generation prompt and any provided input images against the following strict safety guidelines:

1. NSFW/Sexually explicit content is **strictly forbidden**.
2. Real-life public figures (e.g., celebrities, politicians) are **strictly forbidden**. But anonymous, fictional, and historical figures are allowed.
{}

Everything else is allowed: Violence (e.g. fight scenes), weapons, gore...

Analyze the input objectively. If it violates any rule, mark it as unsafe and provide the exact violation reason.

If the prompt is safe and its language is not English, translate it into English without changing its meaning or sacrificing cultural nuances. 
If translation is needed, rendered texts must preserve their original language (e.g. "Merci" est écrit sur la chemise => "Merci" is written on the shirt).
If the input prompt is already in English, do not output translated text.
"""


def get_safety_output_format(generation_type: GenerationType) -> str:
    unsafe_reasons = ["SEXUALLY_EXPLICIT", "REAL_PUBLIC_FIGURE"]
    if generation_type == GenerationType.I2I:
        unsafe_reasons.append("CHILDREN_UNDER_13")

    enum_str = json.dumps(unsafe_reasons)

    return f"""
## Output Format

Return a single JSON object that strictly follows this JSON schema:

```json
{{
  "type": "object",
  "properties": {{
    "is_safe": {{
      "type": "boolean",
      "description": "Indicates whether the generation request is completely safe based on the Safety Guidelines."
    }},
    "unsafe_reason": {{
      "type": "string",
      "enum": {enum_str},
      "description": "The reason why the generation is not safe. Should only be provided when is_safe is false."
    }},
    "translated_prompt": {{
      "type": "string",
      "description":  "Should only be provided when the user input prompt is not in English and is_safe is true."
    }}
  }},
  "required": ["is_safe"]
}}
```
No additional text, only the json output.
"""


class SafetyViolationReason(str, Enum):
    SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
    REAL_PUBLIC_FIGURE = "REAL_PUBLIC_FIGURE"
    CHILDREN_UNDER_13 = "CHILDREN_UNDER_13"
    UNKNOWN = "UNKNOWN"

    @property
    @staticmethod
    def values(self):
        return tuple(r.value for r in SafetyViolationReason)


class SafetyViolationError(Exception):
    def __init__(self, reason: SafetyViolationReason):
        self.reason = reason
        super().__init__(f"Safety violation: {reason}")

    def __str__(self):
        return f'SafetyViolationError(reason="{self.reason}")'


# Helper function to clone a DynamicCache with minimal overhead
def clone_cache(cache: Any) -> Any:
    """
    Safely and efficiently clones a DynamicCache to prevent in-place modification
    during generation, with minimal CPU/GPU overhead.
    """
    if cache is None:
        return None
    try:
        new_cache = copy.copy(cache)
        # Handle newer versions of transformers that store states in a list of CacheLayers
        if hasattr(cache, "layers") and cache.layers:
            new_cache.layers = [copy.copy(layer) for layer in cache.layers]
            for layer in new_cache.layers:
                if hasattr(layer, "key_states") and isinstance(layer.key_states, torch.Tensor):
                    layer.key_states = layer.key_states.clone()
                if hasattr(layer, "value_states") and isinstance(layer.value_states, torch.Tensor):
                    layer.value_states = layer.value_states.clone()
        # Handle traditional versions of transformers that use key_cache and value_cache lists
        if hasattr(cache, "key_cache") and cache.key_cache:
            new_cache.key_cache = [t.clone() if isinstance(t, torch.Tensor) else t for t in cache.key_cache]
        if hasattr(cache, "value_cache") and cache.value_cache:
            new_cache.value_cache = [t.clone() if isinstance(t, torch.Tensor) else t for t in cache.value_cache]
        if hasattr(cache, "_seen_tokens"):
            new_cache._seen_tokens = cache._seen_tokens
        return new_cache
    except Exception:
        # Fallback to standard deepcopy in case of any unexpected cache structure modifications
        return copy.deepcopy(cache)


def move_cache_to_device(cache: Any, device: Any) -> Any:
    """
    Moves all internal tensors of a DynamicCache to the target device in-place.
    Supports both traditional and modern Hugging Face cache structure layouts.
    """
    if cache is None:
        return None
    try:
        # Handle newer versions of transformers that store states in list of CacheLayers
        if hasattr(cache, "layers") and cache.layers:
            for layer in cache.layers:
                if hasattr(layer, "key_states") and isinstance(layer.key_states, torch.Tensor):
                    layer.key_states = layer.key_states.to(device)
                if hasattr(layer, "value_states") and isinstance(layer.value_states, torch.Tensor):
                    layer.value_states = layer.value_states.to(device)
        # Handle traditional versions of transformers that use key_cache and value_cache lists
        if hasattr(cache, "key_cache") and cache.key_cache:
            cache.key_cache = [t.to(device) if isinstance(t, torch.Tensor) else t for t in cache.key_cache]
        if hasattr(cache, "value_cache") and cache.value_cache:
            cache.value_cache = [t.to(device) if isinstance(t, torch.Tensor) else t for t in cache.value_cache]
    except Exception as e:
        print(f"[Prefix Cache Warning] Failed to move cache to {device}: {e}")
    return cache


class PromptEnhancer:
    """Serves as a Prompt Enhancer and Safety Guard"""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        attention_backend: str | None = None,
    ):
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.attention_backend = attention_backend
        apply_sgl_kernel_rmsnorm(self.model, Qwen3VLTextRMSNorm)

        free_memory()

        # Lazy in-memory prefix cache mapping: system_prompt_string -> (DynamicCache, input_ids)
        self._prefix_cache: Dict[str, Tuple[DynamicCache, torch.Tensor]] = {}

    def _get_or_create_prefix_cache_from_inputs(
        self, system_prompt: str, inputs: Any
    ) -> Tuple[Any | None, torch.Tensor | None]:
        """
        Lazily gets or computes the KV states for the given system prompt.
        Designed to persist data on CPU to remain safe for Hugging Face ZeroGPU.
        """
        if not system_prompt:
            return None, None

        if system_prompt not in self._prefix_cache:
            try:
                tokenizer = self.processor.tokenizer
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id is None or (isinstance(im_end_id, int) and im_end_id < 0):
                    im_end_id = tokenizer.eos_token_id

                input_ids_list = inputs.input_ids[0].tolist()
                if im_end_id in input_ids_list:
                    prefix_len = input_ids_list.index(im_end_id) + 1

                    system_inputs = {
                        "input_ids": inputs.input_ids[:, :prefix_len],
                    }
                    if "attention_mask" in inputs:
                        system_inputs["attention_mask"] = inputs.attention_mask[:, :prefix_len]

                    # Compute the system prompt KV states on GPU
                    with torch.inference_mode():
                        outputs = self.model(**system_inputs, use_cache=True)
                        cpu_cache = move_cache_to_device(outputs.past_key_values, "cpu")
                        cpu_ids = system_inputs["input_ids"].to("cpu")
                        self._prefix_cache[system_prompt] = (cpu_cache, cpu_ids)
                else:
                    return None, None
            except Exception as e:
                print(f"[Prefix Cache Warning] Failed to compute prefix cache: {e}")
                return None, None

        cached_cache, cached_ids = self._prefix_cache[system_prompt]
        # Clone the CPU cache, then transfer the clean clone copy to the active GPU device
        gpu_cache = move_cache_to_device(clone_cache(cached_cache), self.model.device)
        gpu_ids = cached_ids.to(self.model.device) if cached_ids is not None else None
        return gpu_cache, gpu_ids

    def _optimize_for_token_count(self, token_count: int):
        if self.attention_backend:
            if token_count < 2048:
                self.model.set_attn_implementation("sdpa")
            else:
                selected_attn = self.attention_backend
                self.model.set_attn_implementation(selected_attn)

    @torch.inference_mode()
    def enhance_prompt(self, prompt: str, generation_type: GenerationType, images: list[str] = []) -> str:
        """
        Enhance the given prompt.
        For video prompt enhancement, the first image is treated as first frame and the second (if any), is treated as last frame.
        Throws SafetyViolationError if the prompt is unsafe
        """

        system_prompt = self._get_system_prompt(generation_type, len(images))

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            }
        ]
        messages.append(self._get_user_message(prompt, images))

        inputs = self.processor.apply_chat_template(  # type: ignore
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        self._optimize_for_token_count(inputs.input_ids.shape[-1])

        prefix_cache, system_input_ids = self._get_or_create_prefix_cache_from_inputs(system_prompt, inputs)
        if prefix_cache is not None and system_input_ids is not None:
            prefix_len = system_input_ids.shape[-1]
            if inputs.input_ids.shape[-1] >= prefix_len and torch.equal(
                inputs.input_ids[0, :prefix_len], system_input_ids[0]
            ):
                pass  # prefix_cache is ready to use
            else:
                prefix_cache = None

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            past_key_values=prefix_cache,
            **self._get_gen_params(len(images) > 0),
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        output_text = self.processor.batch_decode(  # type: ignore
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        output_json = self._parse_json(output_text)

        self._ensure_safe(output_json)

        return output_json["enhanced_prompt"]

    @torch.inference_mode()
    def ensure_prompt_safety(self, prompt: str, generation_type: GenerationType, images: list[str] = []) -> str | None:
        """
        Evaluates the safety of the prompt and input images without modifying the prompt.
        Throws a SafetyViolationError if the inputs violate safety guidelines.
        """

        system_prompt = SAFETY_CHECK_SYS_PROMPT.format(
            CHILDREN_UNDER_13_EDITING_GUIDELINE if generation_type == GenerationType.I2I else ""
        )
        system_prompt = system_prompt + get_safety_output_format(generation_type)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]
        messages.append(self._get_user_message(prompt, images))

        inputs = self.processor.apply_chat_template(  # pyrefly: ignore
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        self._optimize_for_token_count(inputs.input_ids.shape[-1])

        prefix_cache, system_input_ids = self._get_or_create_prefix_cache_from_inputs(system_prompt, inputs)
        if prefix_cache is not None and system_input_ids is not None:
            prefix_len = system_input_ids.shape[-1]
            if inputs.input_ids.shape[-1] >= prefix_len and torch.equal(
                inputs.input_ids[0, :prefix_len], system_input_ids[0]
            ):
                pass  # prefix_cache is ready to use
            else:
                prefix_cache = None

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            past_key_values=prefix_cache,
            **self._get_safety_checker_gen_params(len(images) > 0),
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        output_text = self.processor.batch_decode(  # pyrefly: ignore
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        output_json = self._parse_safety_json(output_text)

        self._ensure_safe(output_json)

        if output_json.get("translated_prompt"):
            return output_json["translated_prompt"]

    def _get_user_message(self, prompt: str, images: list[str]) -> dict:
        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt})

        return {"role": "user", "content": user_content}

    def _get_system_prompt(self, generation_type: GenerationType, num_images: int) -> str:
        output_format = get_output_format(generation_type)
        match generation_type:
            case GenerationType.T2I:
                return IMAGE_GENERATION_SYS_PROMPT + output_format
            case GenerationType.I2I:
                assert num_images > 0, "At least one image is required for image editing"
                return IMAGE_EDITING_SYS_PROMPT + output_format
            case GenerationType.T2V:
                return TEXT_TO_VIDEO_SYS_PROMPT + output_format
            case GenerationType.I2V:
                assert num_images > 0, "At least one image is required for image to video"
                if num_images == 2:
                    return FIRST_LAST_FRAME_SYS_PROMPT + output_format
                else:
                    return IMAGE_TO_VIDEO_SYS_PROMPT + output_format
            case _:
                raise Exception(f"Unsupported generation type: {generation_type}")

    def _get_gen_params(self, has_images) -> dict:
        """
        These allow more diverse outputs compared to self._get_safety_checker_gen_params
        """
        if has_images:
            return {
                "do_sample": True,
                "temperature": 0.9,
                "top_p": 1.0,  # Handled dynamically by min_p
                "top_k": 0,  # Disabled to prevent early truncation
                "min_p": 0.05,  # Only tokens with >= 5% of the top token's prob are kept
                "repetition_penalty": 1.12,  # Prevents repetitive cliches
            }
        else:
            return {
                "do_sample": True,
                "temperature": 1.1,
                "top_p": 1.0,
                "top_k": 0,  # Disabled to prevent early truncation
                "min_p": 0.06,  # Slightly more selective for higher temperature
                "repetition_penalty": 1.15,  # Encourages more diverse stylistic synonyms
            }

    def _get_safety_checker_gen_params(self, has_images) -> dict:
        """
        Returns transformers-compatible generation parameters based on the official recommended settings for this model.
        """
        if has_images:
            return {
                "do_sample": True,
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0.7,
                "repetition_penalty": 1.0,
            }
        else:
            return {
                "do_sample": True,
                "top_p": 1.0,
                "top_k": 40,
                "temperature": 1.0,
                "repetition_penalty": 1.0,
            }

    def _parse_json(self, text: str) -> dict:
        json_data = {}
        try:
            json_data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            # Try finding first { and last }
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if "is_safe" not in json_data and "properties" in json_data:
            print("[PROMPT_ENHANCER] The model my return the full schema instead of the requested json object only")
            json_data = json_data["properties"]

        is_prompt_safe = json_data.get("is_safe")

        if isinstance(is_prompt_safe, str):
            cleaned = is_prompt_safe.lower().strip()
            if cleaned == "true":
                is_prompt_safe = True
            elif cleaned == "false":
                is_prompt_safe = False

            json_data["is_safe"] = is_prompt_safe

        if not isinstance(is_prompt_safe, bool) or is_prompt_safe and not json_data.get("enhanced_prompt", "").strip():
            raise Exception(f"Failed to enhance prompt, got: {text}")

        return json_data

    def _parse_safety_json(self, text: str) -> dict:
        json_data = {}
        try:
            json_data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            # Try finding first { and last }
            if not json_data:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        json_data = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass

        if "is_safe" not in json_data and "properties" in json_data:
            print("[PROMPT_ENHANCER] The model my return the full schema instead of the requested json object only")
            json_data = json_data["properties"]

        is_prompt_safe = json_data.get("is_safe")

        if isinstance(is_prompt_safe, str):
            cleaned = is_prompt_safe.lower().strip()
            if cleaned == "true":
                is_prompt_safe = True
            elif cleaned == "false":
                is_prompt_safe = False

            json_data["is_safe"] = is_prompt_safe

        if not isinstance(is_prompt_safe, bool):
            raise Exception(f"Failed to parse safety check response, got: {text}")

        return json_data

    def _ensure_safe(self, output_json: dict):
        if not output_json["is_safe"]:
            try:
                reason = SafetyViolationReason(output_json.get("unsafe_reason"))
            except ValueError:
                reason = SafetyViolationReason.UNKNOWN

            raise SafetyViolationError(reason)


__all__ = ["PromptEnhancer"]
