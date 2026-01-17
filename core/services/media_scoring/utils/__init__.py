import torch
from transformers import AutoProcessor, AttentionInterface, AutoConfig
from core.services.media_scoring.image_scorer.model import (
    Qwen2VLRewardModelBT as Qwen2VLRewardModelBTImage,
)
from core.services.media_scoring.image_scorer.utils import (
    ModelConfig as ModelConfigImage,
)

from core.services.media_scoring.video_scorer.model import (
    Qwen2VLRewardModelBT as Qwen2VLRewardModelBTVideo,
)
from core.services.media_scoring.video_scorer.utils import (
    ModelConfig as ModelConfigVideo,
)
from .vision_processing import process_vision_info


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
    model_config: ModelConfigImage | ModelConfigVideo,
    model_class: type(Qwen2VLRewardModelBTImage) | type(Qwen2VLRewardModelBTVideo),
    cache_dir=None,
):
    # create processor and set padding
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = model_config.special_tokens
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    config._attn_implementation = attn_backend
    config.torch_dtype = torch.bfloat16
    model_params = {
        "output_dim": model_config.output_dim,
        "reward_token": model_config.reward_token,
        "special_token_ids": special_token_ids,
    }
    if isinstance(model_config, ModelConfigImage):
        model_params.update(
            {
                "rm_head_type": model_config.rm_head_type,
                "rm_head_kwargs": model_config.rm_head_kwargs,
            }
        )
    model = model_class(config, **model_params)

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor


__all__ = [create_model_and_processor, process_vision_info]
