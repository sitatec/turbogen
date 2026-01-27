import torch
from transformers import AttentionInterface
from core.utils.image_utils import image_tensor_to_numpy, image_tensor_to_pil
from core.utils.video_utils import save_video_tensor


attention_backend = "sdpa"

major, minor = torch.cuda.get_device_capability(0)
is_hopper = major == 9

if is_hopper:
    try:
        import flash_attn_3

        attention_backend = "flash_attention_3"
    except ImportError:
        print("FlashAttention-3 not installed. Checking SageAttention.")

if attention_backend == "sdpa":
    try:
        from sageattention import sageattn

        def sage_attention(module, query_states, key_states, value_states, _, **kwargs):
            return sageattn(query_states, key_states, value_states, tensor_layout="HND")

        AttentionInterface.register("sage_attention", sage_attention)
        attention_backend = "sage_attention"
    except ImportError:
        print("SageAttention not installed. Falling back to SDPA.")


__all__ = [
    image_tensor_to_numpy,
    image_tensor_to_pil,
    save_video_tensor,
    attention_backend,
]
