import gc
from contextlib import contextmanager

import torch


@contextmanager
def disable_manual_memory_gc():
    """
    Disable torch.cuda.empty_cache and gc.collect for this context
    """
    orig_gc = gc.collect
    orig_sync = torch.cuda.empty_cache
    gc.collect = lambda *a, **k: 0
    torch.cuda.empty_cache = lambda *a, **k: None
    try:
        yield
    finally:
        gc.collect = orig_gc
        torch.cuda.empty_cache = orig_sync


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
