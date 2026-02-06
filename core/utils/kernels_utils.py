import sys

import torch
import torch.nn as nn
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

    # TODO: check if already installed and skip
    global flash_attn_3_loaded

    if not flash_attn_3_loaded:
        fa3_module = get_kernel("kernels-community/flash-attn3")
        sys.modules["flash_attn_interface"] = fa3_module
        sys.modules["flash_attn3"] = fa3_module
        flash_attn_3_loaded = True
    else:
        print("Flash Attention already loaded, skipping.")


def load_sage_attention():
    # TODO: check if already installed and skip
    global sage_attn_loaded

    if not sage_attn_loaded:
        sage_attn_module = get_kernel("kernels-community/sage-attention")
        sage_attn_module.sageattn_qk_int8_pv_fp16_triton = sage_attn_module.sageattn  # type: ignore
        sys.modules["sageattention"] = sage_attn_module
        sage_attn_loaded = True
    else:
        print("Sage Attention already loaded, skipping.")


def apply_sgl_kernel_rmsnorm(model: nn.Module, rmsnorm_class: type[nn.Module]):
    # Adapted from https://github.com/ModelTC/LightX2V/blob/f76e82c/lightx2v/models/input_encoders/lightllm/qwen25_text_encoder_kernel.py
    print("⚡️ Applying fused RMSNorm kernels from sgl_kernel")
    try:
        from sgl_kernel.elementwise import rmsnorm as optimized_rmsnorm
    except ImportError as e:
        print(f"✗ Failed to import sgl_kernel: {e}. RMSNorm optimization not applied.")
        return

    class OptimizedRMSNormWrapper(nn.Module):
        def __init__(self, original_norm, kernel_fn):
            super().__init__()
            self.weight = original_norm.weight
            self.variance_epsilon = original_norm.variance_epsilon
            self.kernel_fn = kernel_fn

        def forward(self, hidden_states):
            orig_shape = hidden_states.shape
            # Reshape to (-1, hidden_dim) as sgl_kernel expects 2D
            x_2d = hidden_states.view(-1, orig_shape[-1])
            out_2d = self.kernel_fn(x_2d, self.weight, self.variance_epsilon)
            return out_2d.view(orig_shape)

    replaced_count = 0

    def replace_rmsnorm_recursive(module: nn.Module, parent_name=""):
        nonlocal replaced_count
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, rmsnorm_class):
                optimized = OptimizedRMSNormWrapper(child, optimized_rmsnorm)
                setattr(module, name, optimized)
                replaced_count += 1
            else:
                replace_rmsnorm_recursive(child, full_name)

    replace_rmsnorm_recursive(model)

    print(f"✓ Replaced {replaced_count} RMSNorm layers with sgl_kernel version")
