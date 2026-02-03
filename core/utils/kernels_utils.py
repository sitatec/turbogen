import sys

import torch
from kernels import get_kernel

flash_attn_3_loaded = False
sage_attn_loaded = False


def get_gpu_major():
    major, _ = torch.cuda.get_device_capability(0)
    return major


def is_hopper_gpu():
    return get_gpu_major() == 9


def load_flash_attention_3(fallback_to_sage_if_not_hopper=True):
    if fallback_to_sage_if_not_hopper and not is_hopper_gpu():
        return load_sage_attention()

    global flash_attn_3_loaded

    if not flash_attn_3_loaded:
        fa3_module = get_kernel("kernels-community/flash-attn3")
        sys.modules["flash_attn_interface"] = fa3_module
        sys.modules["flash_attn3"] = fa3_module
        flash_attn_3_loaded = True
    else:
        print("Flash Attention already loaded, skipping.")


def load_sage_attention():
    global sage_attn_loaded

    if not sage_attn_loaded:
        sage_attn_module = get_kernel("kernels-community/sage-attention")
        sage_attn_module.sageattn_qk_int8_pv_fp16_triton = sage_attn_module.sageattn  # type: ignore
        sys.modules["sageattention"] = sage_attn_module
        sage_attn_loaded = True
    else:
        print("Sage Attention already loaded, skipping.")
