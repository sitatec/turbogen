from collections.abc import Mapping

import torch
import huggingface_hub
from core.services.media_scoring.vision_processing import process_vision_info
from core.services.media_scoring.image_scorer.utils import (
    create_model_and_processor,
    prompt_with_special_token,
    prompt_without_special_token,
    INSTRUCTION,
)


class ImageScorer:
    def __init__(
        self,
        model_config={
            "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
            "use_special_tokens": True,
            "output_dim": 2,
            "reward_token": "special",
            "rm_head_type": "ranknet",
            "rm_head_kwargs": {},
        },
        checkpoint_path=None,
        device="cuda",
    ):
        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download(
                "MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model"
            )

        model, processor = create_model_and_processor(model_config)

        self.device = device
        self.use_special_tokens = model_config["use_special_tokens"]

        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch

            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(
            attention_mask, padding, "constant", 0
        )

        return sequences_padded, attention_mask_padded

    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def _prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(
                message_list, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.inference_mode()
    def score(self, prompts: list[str], image_or_paths: torch.Tensor | list[str]):
        batch = self._prepare_batch(image_or_paths, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]
        scores = [reward[0].item() for reward in rewards]  # Extract mu values
        return scores


__all__ = [ImageScorer]
