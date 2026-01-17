import os
from collections.abc import Mapping

import torch
import huggingface_hub
from core.services.media_scoring.utils import process_vision_info
from core.services.media_scoring.video_scorer.utils import build_prompt
from core.services.media_scoring.video_scorer.utils import (
    load_configs_from_json,
    ModelConfig,
    DataConfig,
)
from core.services.media_scoring.utils import create_model_and_processor
from core.services.media_scoring.video_scorer.model import Qwen2VLRewardModelBT


class VideoScorer:
    def __init__(
        self,
        model_path: str,
        device="cuda",
    ):
        if not os.path.exists(model_path):
            model_path = huggingface_hub.snapshot_download(
                "KlingTeam/VideoReward", local_dir=model_path, revision="2e08683"
            )

        config_path = os.path.join(model_path, "model_config.json")
        model_config, data_config, inference_config = load_configs_from_json(
            config_path
        )
        model_config = ModelConfig(**model_config)
        model, processor = create_model_and_processor(
            model_config=model_config, model_class=Qwen2VLRewardModelBT
        )

        checkpoint_path = os.path.join(model_path, "checkpoint-11352")
        full_ckpt = os.path.join(checkpoint_path, "model.pth")
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        model.load_state_dict(model_state_dict)
        model.eval()

        self.data_config = DataConfig(**data_config)
        self.device = device
        self.inference_config = inference_config
        self.processor = processor
        self.model = model
        self.model.to(self.device)

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward["VQ"] = (
                reward["VQ"] - self.inference_config["VQ_mean"]
            ) / self.inference_config["VQ_std"]
            reward["MQ"] = (
                reward["MQ"] - self.inference_config["MQ_mean"]
            ) / self.inference_config["MQ_std"]
            reward["TA"] = (
                reward["TA"] - self.inference_config["TA_mean"]
            ) / self.inference_config["TA_std"]
            return reward

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
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
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

    def prepare_batch(
        self,
        videos_or_paths,
        prompts,
        fps=None,
        num_frames=None,
        max_pixels=None,
    ):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = (
            self.data_config.max_frame_pixels if max_pixels is None else max_pixels
        )

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_or_path,
                                "max_pixels": max_pixels,
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt,
                                    self.data_config.eval_dim,
                                    self.data_config.prompt_template_type,
                                ),
                            },
                        ],
                    },
                ]
                for video_or_path, prompt in zip(videos_or_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_or_path,
                                "max_pixels": max_pixels,
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt,
                                    self.data_config.eval_dim,
                                    self.data_config.prompt_template_type,
                                ),
                            },
                        ],
                    },
                ]
                for video_or_path, prompt in zip(videos_or_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(
                chat_data, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def score(
        self,
        videos_or_paths: torch.Tensor | list[str],
        prompts: list[str] | None = None,
        fps=None,
        num_frames=None,
        max_pixels=None,
        use_norm=True,
    ):
        """
        Inputs:
            videos_or_paths: List[str], B videos or paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, (
            "fps and num_frames cannot be set at the same time."
        )
        prompts = prompts or [
            "A video with excellent visual quality, motion quality, and aesthetic appeal."
        ] * len(videos_or_paths)
        batch = self.prepare_batch(
            videos_or_paths, prompts, fps, num_frames, max_pixels
        )
        rewards = self.model(return_dict=True, **batch)["logits"]

        rewards = [
            {"VQ": reward[0].item(), "MQ": reward[1].item(), "TA": reward[2].item()}
            for reward in rewards
        ]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]["Overall"] = (
                rewards[i]["VQ"] + rewards[i]["MQ"] + rewards[i]["TA"]
            )

        return rewards


__all__ = [VideoScorer]
