import torch
from ..diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from ..diffusers_helper.memory import (
    cpu,
    gpu,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
)
from ..diffusers_helper.hunyuan import (
    vae_decode_with_tiling,
)


# =============================================================================
# GENERATION LOOP COORDINATOR
# =============================================================================
class GenerationLoopCoordinator:
    """Coordinates the main generation loop."""

    def __init__(
        self,
        generator,
        high_vram: bool,
        ultra_low_vram: bool,
        settings: dict,
        model_type: str,
    ):
        self.generator = generator
        self.high_vram = high_vram
        self.ultra_low_vram = ultra_low_vram
        self.settings = settings
        self.model_type = model_type

    def run_section(
        self,
        section_params: dict,
    ) -> torch.Tensor:
        """
        Run generation for a single section.

        Args:
            section_params: Dictionary containing all parameters for this section
                - latent_padding_size, latent_window_size, resolutionW/H
                - prompt embeddings, start_latent, history_latents
                - cfg, gs, rs, steps, seed, etc.

        Returns:
            Generated latents for this section
        """
        # Prepare indices
        (
            clean_latent_indices,
            latent_indices,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
        ) = self.generator.prepare_indices(
            section_params["latent_padding_size"], section_params["latent_window_size"]
        )

        # Prepare clean latents
        clean_latents, clean_latents_2x, clean_latents_4x = (
            self.generator.prepare_clean_latents(
                section_params["start_latent"], section_params["history_latents"]
            )
        )

        # Manage model loading for low VRAM
        if not self.high_vram:
            self._load_transformer_for_generation(section_params.get("selected_loras"))

        # Run sampling
        generated_latents = sample_hunyuan(
            transformer=self.generator.transformer,
            width=section_params["resolutionW"],
            height=section_params["resolutionH"],
            frames=section_params["num_frames"],
            real_guidance_scale=section_params["cfg"],
            distilled_guidance_scale=section_params["gs"],
            guidance_rescale=section_params["rs"],
            num_inference_steps=section_params["steps"],
            generator=section_params["random_generator"],
            prompt_embeds=section_params["llama_vec"],
            prompt_embeds_mask=section_params["llama_attention_mask"],
            prompt_poolers=section_params["clip_l_pooler"],
            negative_prompt_embeds=section_params["llama_vec_n"],
            negative_prompt_embeds_mask=section_params["llama_attention_mask_n"],
            negative_prompt_poolers=section_params["clip_l_pooler_n"],
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=section_params["image_encoder_hidden_state"],
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=section_params.get("callback"),
        )

        # Handle last section special case
        if (
            self.model_type in ("Original", "Original with Endframe")
            and section_params.get("has_input_image")
            and section_params.get("is_last_section")
        ):
            generated_latents = torch.cat(
                [
                    section_params["start_latent"].to(generated_latents),
                    generated_latents,
                ],
                dim=2,
            )

        # Offload transformer for low VRAM
        if not self.high_vram:
            self._offload_transformer_after_generation(
                section_params.get("selected_loras")
            )

        return generated_latents

    def decode_and_update_pixels(
        self,
        history_latents: torch.Tensor,
        history_pixels: torch.Tensor,
        total_generated_latent_frames: int,
        latent_window_size: int,
        is_last_section: bool,
        has_input_image: bool,
        vae,
    ):
        """Decode latents and update history pixels."""
        real_history_latents = self.generator.get_real_history_latents(
            history_latents, total_generated_latent_frames
        )

        if history_pixels is None:
            history_pixels = vae_decode_with_tiling(
                real_history_latents, vae, low_vram_tiling_enabled=self.ultra_low_vram
            ).cpu()
        else:
            section_latent_frames = (
                (latent_window_size * 2 + 1)
                if self.model_type in ("Original", "Original with Endframe")
                and has_input_image
                and is_last_section
                else self.generator.get_section_latent_frames(
                    latent_window_size, is_last_section
                )
            )

            current_pixels = self.generator.get_current_pixels(
                real_history_latents,
                section_latent_frames,
                vae,
                low_vram_tiling_enabled=self.ultra_low_vram,
            )

            overlapped_frames = min(
                latent_window_size * 4 - 3,
                history_pixels.shape[2],
                current_pixels.shape[2],
            )

            history_pixels = self.generator.update_history_pixels(
                history_pixels, current_pixels, overlapped_frames
            )

        return real_history_latents, history_pixels

    def _load_transformer_for_generation(self, selected_loras):
        """Load transformer and LoRAs to GPU."""
        from ..diffusers_helper.memory import unload_complete_models

        # Unload other models
        unload_complete_models()

        # Load transformer
        move_model_to_device_with_memory_preservation(
            self.generator.transformer,
            target_device=gpu,
            preserved_memory_gb=self.settings.get("gpu_memory_preservation"),
        )

        # Move LoRAs if needed
        if selected_loras:
            self.generator.move_lora_adapters_to_device(gpu)

    def _offload_transformer_after_generation(self, selected_loras):
        """Offload transformer and LoRAs from GPU."""
        if selected_loras:
            self.generator.move_lora_adapters_to_device(cpu)

        offload_model_from_device_for_memory_preservation(
            self.generator.transformer,
            target_device=gpu,
            preserved_memory_gb=8,
        )
