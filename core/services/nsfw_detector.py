from enum import Enum
from typing import List, Dict, Union

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification


class NsfwLevel(Enum):
    """Enum for NSFW content levels."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self):
        return {
            NsfwLevel.SAFE: 0,
            NsfwLevel.LOW: 1,
            NsfwLevel.MEDIUM: 2,
            NsfwLevel.HIGH: 3,
        }[self]


InputType = Union[
    Image.Image,
    torch.Tensor,
    np.ndarray,
    List[Image.Image | torch.Tensor | np.ndarray],
]


class NsfwDetector:
    """
    NSFW image detector using an INT8 ONNX model on CPU.

    Supported input types:
    - PIL.Image
    - torch.Tensor (CHW or HWC)
    - numpy.ndarray (HWC, RGB or BGR)
    - List of any of the above (can combine the 3 types in the list: [pil, numpy, torch,...])
    """

    REPO_ID = "Freepik/nsfw_image_detector"

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        self.model = AutoModelForImageClassification.from_pretrained(
            self.REPO_ID,
            torch_dtype=torch_dtype,
        ).to(device)

        self.input_size = (448, 448)
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        self.idx_to_label = {
            0: NsfwLevel.SAFE,
            1: NsfwLevel.LOW,
            2: NsfwLevel.MEDIUM,
            3: NsfwLevel.HIGH,
        }

    def _preprocess_one(
        self, img: Image.Image | torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """
        Preprocess a single image into CHW float32 NumPy tensor.
        """
        if isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))
        elif isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))  # to HWC RGB

        # Ensure RGB
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype != np.uint8 else img

        image_height, image_width, _ = img.shape
        scale = max(self.input_size[0] / image_height, self.input_size[1] / image_width)
        resized = cv2.resize(
            img,
            (int(image_width * scale), int(image_height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        cropped = self._center_crop_safe(resized, self.input_size[0])

        # Normalize
        cropped = cropped.astype(np.float32) / 255.0
        cropped = (cropped - self.mean) / self.std

        # HWC → CHW
        return np.transpose(cropped, (2, 0, 1))

    def _center_crop_safe(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w, _ = img.shape

        pad_h = max(0, size - h)
        pad_w = max(0, size - w)

        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(
                img,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0],
            )
            h, w, _ = img.shape

        y0 = (h - size) // 2
        x0 = (w - size) // 2

        return img[y0 : y0 + size, x0 : x0 + size]

    def _preprocess(self, images: InputType) -> np.ndarray:
        """
        Prepare inputs for the ONNX model.

        Returns:
            Dict[str, np.ndarray] compatible with ORTModel
        """
        if not isinstance(images, list):
            images = [images]

        return np.stack([self._preprocess_one(img) for img in images], axis=0)

    def predict_probabilities(self, images: InputType) -> List[Dict[NsfwLevel, float]]:
        """
        Predict probability scores for each NSFW level.

        Args:
            images:
                - PIL.Image
                - torch.Tensor
                - numpy.ndarray
                - List of any of the above (can combine the 3 types in the list: [pil, numpy, torch,...])

        Returns:
            A list of dictionaries with probability scores for each level.
        """
        inputs = self._preprocess(images)

        logits = self.model(inputs).logits
        batch_probs = torch.softmax(logits, dim=-1)

        output = []
        for element_probs in batch_probs:
            output_img = {}
            danger_accumulation = torch.scalar_tensor(0.0)

            # We iterate in reverse order to accumulate danger levels from high to low level.
            # This way, even if the model predict for example a probability of 0.8 for HIGH,
            # and 0.2 for MEDIUM, we will still accumulate the probabilities correctly,
            # giving MEDIUM a value of 1, otherwise it would be 0.2 which is not correct since
            # that would mean that the image is fairly safe, which is not the case since HIGH is 0.8.
            for j in range(len(element_probs) - 1, -1, -1):
                danger_accumulation += element_probs[j]
                if j == 0:  # Neutral level does not accumulate danger
                    danger_accumulation = element_probs[j]
                output_img[self.idx_to_label[j]] = danger_accumulation.item()

            output.append(output_img)

        return output

    def get_nsfw_level(self, images: InputType) -> NsfwLevel | List[NsfwLevel]:
        """
        Get the predicted NSFW level(s) for the input image(s).

        Args:
            images:
                - PIL.Image
                - torch.Tensor
                - numpy.ndarray
                - List of any of the above (can combine the 3 types in the list: [pil, numpy, torch,...])
        """
        predictions = self.predict_probabilities(images)

        nsfw_levels: list[NsfwLevel] = []
        for prediction in predictions:
            if prediction[NsfwLevel.HIGH] >= 0.45:
                nsfw_levels.append(NsfwLevel.HIGH)
            elif prediction[NsfwLevel.MEDIUM] >= 0.5:
                nsfw_levels.append(NsfwLevel.MEDIUM)
            elif prediction[NsfwLevel.LOW] >= 0.5:
                nsfw_levels.append(NsfwLevel.LOW)
            else:
                nsfw_levels.append(NsfwLevel.SAFE)

        if not isinstance(images, list):
            return nsfw_levels[0]
        return nsfw_levels

    def is_nsfw(
        self,
        images: InputType,
        threshold_level: NsfwLevel = NsfwLevel.MEDIUM,
        threshold: float = 0.5,
    ) -> bool | List[bool]:
        """
        Check if images contain NSFW content at or above the specified level.

        Args:
            images:
                - PIL.Image
                - torch.Tensor
                - numpy.ndarray
                - List of any of the above (can combine the 3 types in the list: [pil, numpy, torch,...])
        """
        if threshold_level == NsfwLevel.SAFE:
            raise ValueError("threshold_level cannot be NEUTRAL")

        predictions = self.predict_probabilities(images)

        if not isinstance(images, list):
            return predictions[0][threshold_level] >= threshold

        return [pred[threshold_level] >= threshold for pred in predictions]
