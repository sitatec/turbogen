import torch

from transformers import AutoProcessor, AttentionInterface, AutoConfig
from core.services.media_scoring.image_scorer.model import Qwen2VLRewardModelBT


attn_backend = "sdpa"

major, minor = torch.cuda.get_device_capability(0)
is_hopper = major == 9

if is_hopper:
    try:
        import flash_attn_3

        attn_backend = "flash_attention_3"
    except ImportError:
        print("FlashAttention-3 not installed. Checking SageAttention.")

if attn_backend == "sdpa":
    try:
        from sageattention import sageattn

        def sage_attention(module, query_states, key_states, value_states, _, **kwargs):
            return sageattn(query_states, key_states, value_states, tensor_layout="HND")

        AttentionInterface.register("sage_attention", sage_attention)
        attn_backend = "sage_attention"
    except ImportError:
        print("SageAttention not installed. Falling back to SDPA.")


def create_model_and_processor(
    model_config,
    cache_dir=None,
):
    # create processor and set padding
    processor = AutoProcessor.from_pretrained(
        model_config["model_name_or_path"], padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config["use_special_tokens"]:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    config = AutoConfig.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    config._attn_implementation = attn_backend
    config.torch_dtype = torch.bfloat16
    model = Qwen2VLRewardModelBT(
        config,
        output_dim=model_config["output_dim"],
        reward_token=model_config["reward_token"],
        special_token_ids=special_token_ids,
        rm_head_type=model_config["rm_head_type"],
        rm_head_kwargs=model_config["rm_head_kwargs"],
    ).to("cuda")

    if model_config["use_special_tokens"]:
        model.resize_token_embeddings(len(processor.tokenizer))

    model.rm_head.to(torch.float32)

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor


INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

INSTRUCTION_debug = """
{text_prompt}
"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""
