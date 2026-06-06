import os
import sys
import types
from pathlib import Path
import importlib.machinery
from typing import Optional
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core_utils import is_package_installed


flash_attn_loaded = is_package_installed("flash-attn-4")
sage_attn_loaded = is_package_installed("sageattention")


@lru_cache()
def get_gpu_major():
    major, _ = torch.cuda.get_device_capability(0)
    return major


@lru_cache()
def is_hopper_gpu_or_higher():
    return get_gpu_major() >= 9


def load_flash_attention(fallback_to_sage_if_not_hopper=True):
    from kernels import get_kernel

    if fallback_to_sage_if_not_hopper and not is_hopper_gpu_or_higher():
        return load_sage_attention()

    # TODO: check if already installed and skip
    global flash_attn_loaded

    if not flash_attn_loaded:
        fa_module = get_kernel("kernels-community/flash-attn4", version=0, trust_remote_code=True)
        sys.modules["flash_attn.cute"] = fa_module
        flash_attn_loaded = True
    else:
        print("Flash Attention already loaded, skipping.")


def load_sage_attention(register_to_transformers: bool = True):
    from kernels import get_kernel

    # TODO: check if already installed and skip
    global sage_attn_loaded

    if not sage_attn_loaded:
        sage_attn_module = get_kernel("kernels-community/sage-attention", version=2, trust_remote_code=True)
        sage_attn_module.sageattn_qk_int8_pv_fp16_triton = sage_attn_module.sageattn  # type: ignore
        sys.modules["sageattention"] = sage_attn_module
        sage_attn_loaded = True

        if register_to_transformers:
            from transformers import AttentionInterface

            def sage_attention(module, query_states, key_states, value_states, _, **kwargs):
                return sage_attn_module.sageattn(query_states, key_states, value_states, tensor_layout="HND")

            AttentionInterface.register("sage_attention", sage_attention)
    else:
        print("Sage Attention already loaded, skipping.")


def apply_sgl_kernel_rmsnorm(
    model: nn.Module,
    rmsnorm_class: type[nn.Module],
    epsilon_attr_name: str = "variance_epsilon",
    add_to_weight: int | None = None,
):
    # Adapted from https://github.com/ModelTC/LightX2V/blob/f76e82c/lightx2v/models/input_encoders/lightllm/qwen25_text_encoder_kernel.py
    print(
        f"⚡️ Applying fused RMSNorm kernels from sgl_kernel to {rmsnorm_class.__module__}.{rmsnorm_class.__qualname__}"
    )
    try:
        from sgl_kernel.elementwise import rmsnorm as optimized_rmsnorm
    except ImportError as e:
        print(f"✗ Failed to import sgl_kernel: {e}. RMSNorm optimization not applied.")
        return

    class OptimizedRMSNormWrapper(nn.Module):
        def __init__(self, original_norm, kernel_fn):
            super().__init__()
            if add_to_weight:
                self.weight = add_to_weight + original_norm.weight
            else:
                self.weight = original_norm.weight

            self.variance_epsilon = getattr(original_norm, epsilon_attr_name)
            self.kernel_fn = kernel_fn

        def forward(self, hidden_states, gate=None):
            orig_shape = hidden_states.shape
            # Reshape to (-1, hidden_dim) as sgl_kernel expects 2D
            x_2d = hidden_states.view(-1, orig_shape[-1])
            out_2d = self.kernel_fn(x_2d, self.weight, self.variance_epsilon)
            hidden_states = out_2d.view(orig_shape)
            if gate is not None:
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32) * F.silu(gate.to(torch.float32))
                return hidden_states.to(input_dtype)
            return hidden_states

    replaced_count = 0

    def replace_rmsnorm_recursive(module: nn.Module, parent_name=""):
        nonlocal replaced_count
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, rmsnorm_class) and hasattr(child, "weight") and hasattr(child, epsilon_attr_name):
                optimized = OptimizedRMSNormWrapper(child, optimized_rmsnorm)
                setattr(module, name, optimized)
                replaced_count += 1
            else:
                replace_rmsnorm_recursive(child, full_name)

    replace_rmsnorm_recursive(model)

    print(f"✓ Replaced {replaced_count} RMSNorm layers with sgl_kernel version")


def set_jit_cache_dirs(cache_root_dir: Path):
    os.environ["FLASHINFER_CACHE_DIR"] = f"{cache_root_dir}/flashinfer"
    os.environ["TRITON_CACHE_DIR"] = f"{cache_root_dir}/triton"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_root_dir}/inductor"

    # --- FlashAttention 4 / CuTe DSL (If using FA4 or custom CuTe kernels) ---
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(cache_root_dir / "cute_dsl")
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"

    # Caches dynamic JIT C++/CUDA compilations via torch.utils.cpp_extension (Ninja backend)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_root_dir / "torch_extensions")
    # Caches the NVIDIA driver JIT compilation (PTX to device-specific SASS binary)
    os.environ["CUDA_CACHE_PATH"] = str(cache_root_dir / "cuda_nv_cache")
    # Increase CUDA cache limit (e.g., to ~4GB) so compiled kernels aren't evicted
    os.environ["CUDA_CACHE_MAXSIZE"] = "4294967296"

    # SGLang JIT kernels use tvm-ffi to compile and link C++/CUDA on the fly
    os.environ["TVM_FFI_CACHE_DIR"] = str(cache_root_dir / "tvm_ffi")
    # DeepGEMM maybe used by SGLang workloads on Hopper & Blackwell
    os.environ["DG_CACHE_DIR"] = str(cache_root_dir / "deep_gemm")
    os.environ["SGLANG_DG_CACHE_DIR"] = str(cache_root_dir / "deep_gemm")
    # Caches compiled finite state machines (FSM) used for guided JSON or Regex decoding
    os.environ["OUTLINES_CACHE_DIR"] = str(cache_root_dir / "outlines")


def _sgl_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
):
    # Use standard out-of-place cloning for safety during prefill,
    # but check contiguity efficiently.
    out = x if x.is_contiguous() else x.contiguous()
    out = out.clone()

    conv_states_arg = None
    has_initial_state = None
    if initial_states is not None:
        conv_states_arg = initial_states if initial_states.is_contiguous() else initial_states.contiguous()
        has_initial_state = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    elif return_final_states or final_states_out is not None:
        batch, dim, seqlen = x.shape
        width = weight.shape[1]
        conv_states_arg = torch.zeros(batch, dim, width - 1, dtype=x.dtype, device=x.device)

    # Directly pass parameters to bypass standard CPU check validation
    torch.ops.sgl_kernel.causal_conv1d_fwd(
        out,
        weight,
        bias,
        conv_states_arg,
        None,
        None,
        has_initial_state,
        (activation == "silu" or activation == "swish"),
        -1,
    )

    if return_final_states:
        if final_states_out is not None:
            final_states_out.copy_(conv_states_arg)
        else:
            final_states_out = conv_states_arg
        return out, final_states_out

    return out


def _sgl_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
):
    # 1. Mutate directly in-place to avoid clone/copy allocation overhead.
    # 2. Fast-path unsqueeze to 3D only if needed.
    if x.ndim == 2:
        x = x.unsqueeze(-1)

    # 3. Call the kernel directly. Static weights, biases, and states
    # are already contiguous, avoiding redundant check/cast checks.
    torch.ops.sgl_kernel.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        (activation == "silu" or activation == "swish"),
        cache_seqlens,
        conv_state_indices,
        -1,  # pad_slot_id
    )

    return x


def patch_causal_conv1d_with_sgl_kernel(patch_qwen35: bool = True):
    mod = types.ModuleType("causal_conv1d")
    mod.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", None, origin="mocked")
    mod.causal_conv1d_fn = _sgl_causal_conv1d_fn
    mod.causal_conv1d_update = _sgl_causal_conv1d_update

    sys.modules["causal_conv1d"] = mod
    print("✓ Mocked 'causal_conv1d' successfully in sys.modules")

    if patch_qwen35:
        try:
            import transformers.models.qwen3_5.modeling_qwen3_5 as modeling_qwen3_5

            modeling_qwen3_5.causal_conv1d_fn = _sgl_causal_conv1d_fn
            modeling_qwen3_5.causal_conv1d_update = _sgl_causal_conv1d_update
            print("✓ Pre-patched transformers.models.qwen3_5.modeling_qwen3_5.")
        except ImportError:
            pass
