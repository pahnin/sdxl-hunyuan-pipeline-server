import torch
from ..diffusers_helper.memory import (
    gpu,
    offload_model_from_device_for_memory_preservation,
    load_model_as_complete,
)
from ..diffusers_helper.hunyuan import (
    vae_encode,
)
from ..diffusers_helper.clip_vision import hf_clip_vision_encode
import numpy as np


# =============================================================================
# INPUT PROCESSOR
# =============================================================================
class InputProcessor:
    """Handles input image/video processing and encoding."""

    def __init__(self, vae, image_encoder, feature_extractor, high_vram: bool):
        self.vae = vae
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.high_vram = high_vram

    @torch.no_grad()
    def process_input_image(self, input_image_np: np.ndarray):
        """Process input image: convert to tensor and encode with VAE."""
        # Convert to tensor
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # Load VAE if needed
        if not self.high_vram:
            load_model_as_complete(self.vae, target_device=gpu)

        # Encode
        start_latent = vae_encode(input_image_pt, self.vae)

        return start_latent

    @torch.no_grad()
    def encode_clip_vision(self, input_image_np: np.ndarray):
        """Encode image with CLIP vision model."""
        if not self.high_vram:
            load_model_as_complete(self.image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(
            input_image_np, self.feature_extractor, self.image_encoder
        )

        return image_encoder_output.last_hidden_state

    def offload_models(self, gpu_memory_preservation: float):
        """Offload VAE and image encoder after processing."""
        if not self.high_vram:
            offload_model_from_device_for_memory_preservation(
                self.vae,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )
            offload_model_from_device_for_memory_preservation(
                self.image_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )
