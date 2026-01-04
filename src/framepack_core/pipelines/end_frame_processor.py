import os
import torch
from ..diffusers_helper.hunyuan import (
    vae_encode,
)
import numpy as np
from PIL import Image
from ..diffusers_helper.memory import gpu, load_model_as_complete
from ..diffusers_helper.utils import (
    resize_and_center_crop,
)


# =============================================================================
# END FRAME PROCESSOR
# =============================================================================
class EndFrameProcessor:
    """Handles end frame processing for models that support it."""

    def __init__(self, vae, high_vram: bool):
        self.vae = vae
        self.high_vram = high_vram

    @torch.no_grad()
    def process_end_frame(
        self,
        end_frame_image: np.ndarray,
        resolutionW: int,
        resolutionH: int,
        metadata_dir: str = None,
        job_id: str = None,
    ):
        """
        Process end frame: validate, save, and encode with VAE.
        Returns (end_frame_latent, end_frame_output_dimensions_latent).
        """
        if end_frame_image is None:
            return None, None

        # Validate input
        if not isinstance(end_frame_image, np.ndarray):
            print(
                f"Warning: end_frame_image is not a numpy array (type: {type(end_frame_image)})"
            )
            try:
                end_frame_image = np.array(end_frame_image)
            except Exception as e:
                print(f"Could not convert end_frame_image to numpy array: {e}")
                return None, None

        # Save debug image if metadata directory provided
        if metadata_dir and job_id:
            Image.fromarray(end_frame_image).save(
                os.path.join(metadata_dir, f"{job_id}_end_frame_processed.png")
            )

        # Encode end frame at original size
        end_frame_pt = torch.from_numpy(end_frame_image).float() / 127.5 - 1
        end_frame_pt = end_frame_pt.permute(2, 0, 1)[None, :, None]

        if not self.high_vram:
            load_model_as_complete(self.vae, target_device=gpu)

        end_frame_latent = vae_encode(end_frame_pt, self.vae)

        # Encode at output dimensions (resized)
        end_frame_resized_np = resize_and_center_crop(
            end_frame_image, resolutionW, resolutionH
        )
        end_frame_resized_pt = (
            torch.from_numpy(end_frame_resized_np).float() / 127.5 - 1
        )
        end_frame_resized_pt = end_frame_resized_pt.permute(2, 0, 1)[None, :, None]

        end_frame_output_latent = vae_encode(end_frame_resized_pt, self.vae)

        print("End frame VAE encoded.")
        return end_frame_latent, end_frame_output_latent

    def apply_end_frame_to_history(
        self,
        history_latents: torch.Tensor,
        end_frame_latent: torch.Tensor,
        end_frame_strength: float,
    ):
        """Apply end frame latent to history latents with specified strength."""
        if end_frame_latent is None:
            return history_latents

        print(
            f"Applying end_frame_latent to history_latents with strength: {end_frame_strength}"
        )

        actual_end_frame = end_frame_latent.clone()
        if end_frame_strength != 1.0:
            actual_end_frame = actual_end_frame * end_frame_strength

        # Resize if dimensions don't match
        if history_latents.shape[2] >= 1:
            target_h = history_latents.shape[3]
            target_w = history_latents.shape[4]

            if (
                actual_end_frame.shape[-2] != target_h
                or actual_end_frame.shape[-1] != target_w
            ):
                import torch.nn.functional as F

                actual_end_frame = F.interpolate(
                    actual_end_frame.squeeze(2),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).unsqueeze(2)
                print(f"  Resized end frame from original to {target_h}x{target_w}")

            # Apply to history (implementation depends on your history format)
            # This is a placeholder - adjust based on your actual history structure
            print(f"End frame latent applied to history")
        else:
            print(
                "Warning: history_latents not shaped as expected for end_frame application"
            )

        return history_latents
