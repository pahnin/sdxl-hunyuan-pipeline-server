import torch
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers import AutoencoderKLHunyuanVideo
from .diffusers_helper.memory import DynamicSwapInstaller
from .settings import Settings


class HunyuanVideoGenerator:
    """Dedicated generator for HunyuanVideo model operations."""

    def __init__(self, settings: Settings, high_vram: bool, gpu_device):
        self.settings = settings
        self.high_vram = high_vram
        self.gpu = gpu_device
        self.models = {}

    def load_models(self, model_path: str):
        """Load all required HunyuanVideo models."""
        self.models = {
            "text_encoder": LlamaModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=torch.float16
            ).cpu(),
            "text_encoder_2": CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder_2", torch_dtype=torch.float16
            ).cpu(),
            "tokenizer": LlamaTokenizerFast.from_pretrained(
                model_path, subfolder="tokenizer"
            ),
            "tokenizer_2": CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2"
            ),
            "vae": AutoencoderKLHunyuanVideo.from_pretrained(
                model_path, subfolder="vae", torch_dtype=torch.float16
            ).cpu(),
        }
        return self.models

    def configure_models(self):
        """Configure models for inference."""
        for name, model in self.models.items():
            if hasattr(model, "eval"):
                model.eval()
            if hasattr(model, "requires_grad_"):
                model.requires_grad_(False)
            if hasattr(model, "to"):
                model.to(dtype=torch.float16)

        # VAE-specific configuration
        if not self.high_vram:
            self.models["vae"].enable_slicing()
            self.models["vae"].enable_tiling(
                tile_sample_min_height=257,
                tile_sample_min_width=257,
                tile_sample_min_num_frames=6,
                tile_sample_stride_height=129,
                tile_sample_stride_width=129,
                tile_sample_stride_num_frames=5,
            )

    def apply_memory_management(self):
        """Apply dynamic memory management."""
        if not self.high_vram:
            DynamicSwapInstaller.install_model(
                self.models["text_encoder"], device=self.gpu
            )
        else:
            for model in self.models.values():
                if hasattr(model, "to"):
                    model.to(self.gpu)

    def generate_video(
        self,
        start_img,
        end_img,
        prompt,
        n_prompt,
        seed,
        feature_extractor,
        image_encoder,
        **generation_params
    ):
        """Execute video generation with configured models."""
        from .pipelines import worker

        return worker.worker(
            self.high_vram,
            False,  # ultra_low_vram
            self.settings,
            self.models["text_encoder"],
            self.models["text_encoder_2"],
            self.models["tokenizer"],
            self.models["tokenizer_2"],
            self.models["vae"],
            feature_extractor,
            image_encoder,
            {},  # prompt_embedding_cache
            model_type="Original with Endframe",
            input_image=start_img,
            end_frame_image=end_img,
            end_frame_strength=0.9,
            prompt_text=prompt,
            n_prompt=n_prompt,
            seed=seed,
            **generation_params
        )
