import torch
from PIL import Image
import numpy as np


def image_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts an image(s) with the torch.Tensor type to a numpy array"""

    np_image = tensor.float().detach().cpu().numpy() * 255.0
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
    return np_image


def image_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(image_tensor_to_numpy(tensor))
