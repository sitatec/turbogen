import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    CLIPProcessor,
    CLIPModel,
)

from core.services.media_scoring.qwen2_vision_processing import process_vision_info
from core.utils import attention_backend


class ImageScorer:
    PROMPT = (
        "You are doing the image quality assessment task. Here is the question: "
        "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
        "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
    )

    QUESTION_TEMPLATE = "{Question} Please only output the final answer with only one score in <answer> </answer> tags."

    def __init__(self, models_dir: Path):
        self.device = torch.device("cuda")

        self.vq_model, self.vq_processor = self._create_vq_model(models_dir)

        self.clip_model, self.clip_processor = self._create_clip_model(models_dir)

        self.aesthetic_model = self._create_aesthetic_model(models_dir)

    def _create_vq_model(self, models_dir: Path):
        vq_model_path = models_dir / "visual_quality_r1"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vq_model_path,
            dtype="auto",
            attn_implementation=attention_backend,
            device_map=self.device,
        )
        processor = AutoProcessor.from_pretrained(vq_model_path)
        processor.tokenizer.padding_side = "left"

        return model, processor

    def _create_clip_model(self, models_dir: Path):
        clip_model_path = models_dir / "clip-vit-l14"

        clip_model: CLIPModel = CLIPModel.from_pretrained(
            clip_model_path, device_map=self.device
        )
        clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        return clip_model, clip_processor

    def _create_aesthetic_model(self, models_dir: Path):
        model_name = "sac+logos+ava1-l14-linearMSE.pth"
        model_path = models_dir / model_name

        aesthetic_model = AestheticModel(device=self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        aesthetic_model.load_state_dict(state_dict)
        return aesthetic_model

    @torch.inference_mode()
    def score_aesthetic(self, image: Image.Image | torch.Tensor) -> float:
        """Output the aesthetic sore of the image ranging from 0 to 0.1"""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        normalized_features = F.normalize(image_features, p=2, dim=-1)
        score = self.aesthetic_model(normalized_features).clamp(4, 8).item()
        # Rescale aesthetic_score from [4, 8] -> [0, 0.1]
        score = (score - 4.0) * (0.1 / 4)
        return score

    @torch.inference_mode()
    def score_visual_quality(
        self, image: Image.Image | torch.Tensor, attempt=1
    ) -> float:
        """Output the overall visual quality sore of the image ranging from 0 to 5"""
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": self.QUESTION_TEMPLATE.format(Question=self.PROMPT),
                    },
                ],
            }
        ]

        text = self.vq_processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.vq_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.vq_model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=2048,
            do_sample=True,
            top_k=50,
            top_p=1,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = self.vq_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        try:
            model_output_matches = re.findall(
                r"<answer>(.*?)</answer>", batch_output_text[0], re.DOTALL
            )
            model_answer = (
                model_output_matches[-1].strip()
                if model_output_matches
                else batch_output_text[0].strip()
            )
            # pyrefly: ignore
            score = float(re.search(r"\d+(\.\d+)?", model_answer).group())
            return score
        except Exception as e:
            print("================= Scoring failed, retrying. =================")

            if attempt >= 3:
                raise e

            return self.score_visual_quality(image, attempt=attempt + 1)

    def score(self, image: Image.Image | torch.Tensor) -> float:
        visual_quality_score = self.score_visual_quality(image)
        aesthetic_score = self.score_aesthetic(image)

        combined_score = visual_quality_score + aesthetic_score  # [0, 5.1]
        # Rescale combined scores from [0, 5.1] -> [0, 10]
        combined_score *= 10.0 / 5.1

        return combined_score


class AestheticModel(nn.Module):
    def __init__(self, input_size=768, device="cuda", xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

        self.to(device).eval()

    def forward(self, x: torch.Tensor):
        return self.layers(x.to(device=self.device, dtype=torch.float))
