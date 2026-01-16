import io
import json
import base64

import cv2
import torch
import piexif
import numpy as np
from PIL import Image
from fast_thumbhash import rgba_to_thumb_hash


def image_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts an image(s) with the torch.Tensor type to a numpy array"""

    np_image = tensor.float().detach().cpu().numpy() * 255.0
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
    return np_image


def image_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(image_tensor_to_numpy(tensor))


def convert_to_webp_with_metadata(
    image: np.ndarray | Image.Image,
    metadata: bytes | dict | None,
    quality: int = 100,
    output_path: str | None = None,
) -> bytes | None:
    """
    Convert the given numpy or PIL image, convert it to webp with the exif data.
    If output_path is provided, save it to that path otherwise return the image bytes
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if isinstance(metadata, dict):
        metadata = create_exif_data(metadata)

    if output_path:
        image.save(output_path, format="WEBP", quality=quality, exif=metadata)
    else:
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="WEBP", quality=quality, exif=metadata)
        return output_buffer.getvalue()


def create_exif_data(metadata: dict) -> bytes:
    metadata = metadata.copy()
    # Create separate dictionaries for ImageIFD (0th) and ExifIFD
    ifd_data = {}
    exif_data = {}

    # ImageIFD tags (basic image info)
    if "description" in metadata:
        ifd_data[piexif.ImageIFD.ImageDescription] = metadata["description"].encode(
            "utf-8"
        )
        del metadata["prompt"]

    if "artist" in metadata:
        ifd_data[piexif.ImageIFD.ImageDescription] = metadata["artist"].encode("utf-8")
        del metadata["artist"]

    if "software" in metadata:
        ifd_data[piexif.ImageIFD.ImageDescription] = metadata["software"].encode(
            "utf-8"
        )
        del metadata["software"]

    metadata_str = json.dumps(metadata)
    exif_data[piexif.ExifIFD.UserComment] = metadata_str.encode("utf-8")

    # Assemble the EXIF data dictionary with the correct structure
    return piexif.dump(
        {
            "0th": ifd_data,
            "Exif": exif_data,
        }
    )


def generate_thumbhash(image: np.ndarray) -> str:
    """
    Generates a thumbhash for the given image.
    """

    # Thumbhash encoder expect max 100x100 dimensions
    image_height, image_width = image.shape[:2]
    scale = min(100 / image_width, 100 / image_height)
    new_w, new_h = int(image_width * scale), int(image_height * scale)
    thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    thumb_height, thumb_width = thumbnail.shape[:2]
    rgba_image = np.full((thumb_height, thumb_width, 4), 255, dtype=np.uint8)
    rgba_image[:, :, :3] = thumbnail  # Copy RGB channels

    thumb_hash_bytes = rgba_to_thumb_hash(
        width=thumb_width, height=thumb_height, rgba=rgba_image.tobytes()
    )
    return base64.b64encode(bytes(thumb_hash_bytes)).decode()
