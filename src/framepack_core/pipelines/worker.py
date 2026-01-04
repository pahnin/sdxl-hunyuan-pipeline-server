"""
Worker module for video generation - refactored into focused components.
"""

import os
import traceback
import torch
from pathlib import Path
from ..diffusers_helper.models.mag_cache import MagCache
from ..diffusers_helper.utils import (
    save_bcthw_as_mp4,
    generate_timestamp,
)
from ..diffusers_helper.memory import (
    gpu,
    fake_diffusers_current_device,
    unload_complete_models,
    load_model_as_complete,
)


from ..prompt_handler import parse_timestamped_prompt
from ..generators import create_model_generator
from ..llm_captioner import unload_captioning_model
from ..llm_enhancer import unload_enhancing_model
from .lora_manager import LoRAManager
from .prompt_encoder import PromptEncoder
from .input_processor import InputProcessor
from .end_frame_processor import EndFrameProcessor
from .prompt_blender import PromptBlender
from .progress_tracker import ProgressTracker
from .generation_loop_coordinator import GenerationLoopCoordinator
from .cleanup_manager import CleanupManager

from . import create_pipeline


# =============================================================================
# MAIN WORKER FUNCTION
# =============================================================================
@torch.no_grad()
def worker(
    high_vram,
    ultra_low_vram,
    settings,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    vae,
    feature_extractor,
    image_encoder,
    prompt_embedding_cache,
    model_type,
    input_image,
    end_frame_image,
    end_frame_strength,
    prompt_text,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    use_magcache,
    magcache_threshold,
    magcache_max_consecutive_skips,
    magcache_retention_ratio,
    blend_sections,
    latent_type,
    selected_loras,
    has_input_image,
    lora_values=None,
    job_stream=None,
    output_dir=None,
    metadata_dir=None,
    input_files_dir=None,
    input_image_path=None,
    end_frame_image_path=None,
    resolutionW=640,
    resolutionH=640,
    lora_loaded_names=[],
    input_video=None,
    combine_with_source=None,
    num_cleaned_frames=5,
    save_metadata_checked=True,
):
    """
    Worker function for video generation - refactored and cleaned up.
    """
    # Initialize
    random_generator = torch.Generator("cpu").manual_seed(seed)
    job_id = generate_timestamp()
    generator = None

    # Cleanup ML models
    unload_enhancing_model()
    unload_captioning_model()

    # Filter LoRAs
    selected_loras = LoRAManager.filter_dummy_loras(selected_loras)
    print(f"Worker: Selected LoRAs: {selected_loras}")

    # Calculate sections
    total_latent_sections = max(
        1, round((total_second_length * 30) / (latent_window_size * 4))
    )
    print(f"Total latent sections: {total_latent_sections}")
    print(
        f"Expected frames: {total_second_length * 30}, Frames per section: {latent_window_size * 4}"
    )

    # Parse prompts
    prompt_sections = parse_timestamped_prompt(
        prompt_text, total_second_length, latent_window_size, model_type
    )

    try:
        # Create pipeline
        pipeline = create_pipeline(
            model_type, settings, output_dir, metadata_dir, input_files_dir, high_vram
        )

        # Prepare job parameters
        job_params = pipeline.prepare_parameters(
            {
                "model_type": model_type,
                "input_image": input_image,
                "end_frame_image": end_frame_image,
                "end_frame_strength": end_frame_strength,
                "prompt_text": prompt_text,
                "n_prompt": n_prompt,
                "seed": seed,
                "total_second_length": total_second_length,
                "latent_window_size": latent_window_size,
                "steps": steps,
                "cfg": cfg,
                "gs": gs,
                "rs": rs,
                "blend_sections": blend_sections,
                "latent_type": latent_type,
                "use_magcache": use_magcache,
                "magcache_threshold": magcache_threshold,
                "magcache_max_consecutive_skips": magcache_max_consecutive_skips,
                "magcache_retention_ratio": magcache_retention_ratio,
                "selected_loras": selected_loras,
                "has_input_image": has_input_image,
                "lora_values": lora_values,
                "resolutionW": resolutionW,
                "resolutionH": resolutionH,
                "lora_loaded_names": lora_loaded_names,
                "input_image_path": input_image_path,
                "end_frame_image_path": end_frame_image_path,
                "combine_with_source": combine_with_source,
                "num_cleaned_frames": num_cleaned_frames,
                "save_metadata_checked": save_metadata_checked,
                "output_dir": output_dir,
                "metadata_dir": metadata_dir,
            }
        )

        # Validate
        is_valid, error_message = pipeline.validate_parameters(job_params)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_message}")

        # Create generator
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae)

        generator = create_model_generator(
            model_type,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            high_vram=high_vram,
            prompt_embedding_cache=prompt_embedding_cache,
            offline=True,
            settings=settings,
        )

        generator.load_model()
        generator.unload_loras()

        # Preprocess inputs
        processed_inputs = pipeline.preprocess_inputs(job_params)
        job_params.update(processed_inputs)

        # Save metadata
        if (
            settings.get("save_metadata")
            and job_params.get("save_metadata_checked", True)
            and job_params.get("input_image") is not None
        ):
            try:
                from .metadata_utils import save_job_start_image

                save_job_start_image(job_params, job_id, settings)
                print(f"Saved metadata and starting image for job {job_id}")
            except Exception as e:
                print(f"Error saving metadata: {e}")
                traceback.print_exc()

        # === PROMPT ENCODING ===
        prompt_encoder = PromptEncoder(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt_embedding_cache
        )

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        # Encode all unique prompts
        unique_prompts = []
        for section in prompt_sections:
            if section.prompt not in unique_prompts:
                unique_prompts.append(section.prompt)

        encoded_prompts = prompt_encoder.encode_multiple_prompts(unique_prompts, gpu)

        # Encode negative prompt
        llama_vec_n, llama_attention_mask_n, clip_l_pooler_n = (
            prompt_encoder.encode_negative_prompt(
                n_prompt, cfg, prompt_sections[0].prompt, gpu
            )
        )

        # === INPUT PROCESSING ===
        input_processor = InputProcessor(
            vae, image_encoder, feature_extractor, high_vram
        )

        if input_image is None:
            start_latent = torch.randn(
                (1, 16, 1, resolutionH // 8, resolutionW // 8),
                generator=random_generator,
                device=random_generator.device,
            ).to(device=gpu, dtype=torch.float32)
            import numpy as np

            black_image_np = np.zeros((resolutionH, resolutionW, 3), dtype=np.uint8)
            image_encoder_hidden_state = input_processor.encode_clip_vision(
                black_image_np
            )
        else:
            start_latent = input_processor.process_input_image(input_image)
            image_encoder_hidden_state = input_processor.encode_clip_vision(input_image)

        # === END FRAME PROCESSING ===
        end_frame_latent = None
        if (
            model_type in ("Original with Endframe", "Video")
            and job_params.get("end_frame_image") is not None
        ):
            end_frame_processor = EndFrameProcessor(vae, high_vram)
            end_frame_latent, _ = end_frame_processor.process_end_frame(
                end_frame_image,
                resolutionW,
                resolutionH,
                metadata_dir if settings.get("save_metadata") else None,
                job_id,
            )

        # Offload VAE and image encoder
        input_processor.offload_models(settings.get("gpu_memory_preservation"))

        # === DTYPE CONVERSION ===
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            encoded_prompts[prompt_key] = (
                llama_vec.to(generator.transformer.dtype),
                llama_attention_mask,
                clip_l_pooler.to(generator.transformer.dtype),
            )

        llama_vec_n = llama_vec_n.to(generator.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(generator.transformer.dtype)
        image_encoder_hidden_state = image_encoder_hidden_state.to(
            generator.transformer.dtype
        )

        # === SETUP GENERATION ===
        num_frames = latent_window_size * 4 - 3
        history_latents = generator.prepare_history_latents(resolutionH, resolutionW)
        total_generated_latent_frames = 0
        history_pixels = None
        latent_paddings = generator.get_latent_paddings(total_latent_sections)

        # Load LoRAs
        if selected_loras:
            generator.load_loras(
                selected_loras,
                settings.get("lora_dir"),
                lora_loaded_names,
                lora_values,
            )

        # Setup progress tracking
        progress_tracker = ProgressTracker(
            total_latent_sections * steps, total_second_length, job_stream=job_stream
        )

        # Setup prompt blending
        prompt_blender = PromptBlender(prompt_sections, encoded_prompts, blend_sections)

        # Setup MagCache
        magcache = MagCache(
            model_family="Original",
            height=resolutionH,
            width=resolutionW,
            num_steps=steps,
            is_calibrating=False,
            threshold=magcache_threshold,
            max_consectutive_skips=magcache_max_consecutive_skips,
            retention_ratio=magcache_retention_ratio,
        )
        generator.transformer.initialize_teacache(enable_teacache=False)
        generator.transformer.install_magcache(magcache)

        # Setup generation coordinator
        coordinator = GenerationLoopCoordinator(
            generator, high_vram, ultra_low_vram, settings, model_type
        )

        # === MAIN GENERATION LOOP ===
        section_idx = 0

        for i_section_loop, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            # Calculate time position
            current_time_position = max(
                0.01, (total_generated_latent_frames * 4 - 3) / 30
            )
            original_time_position = max(0, total_second_length - current_time_position)

            # Get prompt for this section (with blending)
            llama_vec, llama_attention_mask, clip_l_pooler = (
                prompt_blender.get_prompt_for_section(
                    section_idx, current_time_position
                )
            )

            print(
                f"Section {section_idx+1}/{total_latent_sections}: "
                f"time={current_time_position:.2f}s, padding={latent_padding_size}, last={is_last_section}"
            )

            # Apply end frame to history (first section only)
            if (
                model_type == "Original with Endframe"
                and i_section_loop == 0
                and end_frame_latent is not None
            ):
                end_frame_processor = EndFrameProcessor(vae, high_vram)
                history_latents = end_frame_processor.apply_end_frame_to_history(
                    history_latents, end_frame_latent, end_frame_strength
                )

            # Run generation for this section
            section_params = {
                "latent_padding_size": latent_padding_size,
                "latent_window_size": latent_window_size,
                "resolutionW": resolutionW,
                "resolutionH": resolutionH,
                "start_latent": start_latent,
                "history_latents": history_latents,
                "llama_vec": llama_vec,
                "llama_attention_mask": llama_attention_mask,
                "clip_l_pooler": clip_l_pooler,
                "llama_vec_n": llama_vec_n,
                "llama_attention_mask_n": llama_attention_mask_n,
                "clip_l_pooler_n": clip_l_pooler_n,
                "image_encoder_hidden_state": image_encoder_hidden_state,
                "cfg": cfg,
                "gs": gs,
                "rs": rs,
                "steps": steps,
                "num_frames": num_frames,
                "random_generator": random_generator,
                "selected_loras": selected_loras,
                "has_input_image": has_input_image,
                "is_last_section": is_last_section,
                "callback": progress_tracker.create_callback(
                    total_generated_latent_frames
                ),
            }

            generated_latents = coordinator.run_section(section_params)

            # Update history
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = generator.update_history_latents(
                history_latents, generated_latents
            )

            # Decode and save
            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents, history_pixels = coordinator.decode_and_update_pixels(
                history_latents,
                history_pixels,
                total_generated_latent_frames,
                latent_window_size,
                is_last_section,
                has_input_image,
                vae,
            )

            if not high_vram:
                unload_complete_models()

            # Save video
            output_filename = os.path.join(
                output_dir, f"{job_id}_{total_generated_latent_frames}.mp4"
            )
            save_bcthw_as_mp4(
                history_pixels, output_filename, fps=30, crf=settings.get("mp4_crf")
            )
            print(f"Saved: {output_filename}, pixels shape: {history_pixels.shape}")

            if is_last_section:
                break

            section_idx += 1

        # === CLEANUP ===
        magcache = generator.transformer.magcache
        print(
            f"MagCache: {100.0 * magcache.total_cache_hits / magcache.total_cache_requests:.2f}% "
            f"({magcache.total_cache_hits}/{magcache.total_cache_requests} steps)"
        )
        generator.transformer.uninstall_magcache()

        preview_filename = f"{job_id}_{total_generated_latent_frames}_end_frame.png"

        #         # Return info for DB / client
        # result = {
        # "video_filename": os.path.basename(output_filename),
        # "preview_image": preview_filename,
        # }

        # In your generation loop where you save files
        output_filename = os.path.join(
            output_dir, f"{job_id}_{total_generated_latent_frames}.mp4"
        )
        save_bcthw_as_mp4(
            history_pixels, output_filename, fps=30, crf=settings.get("mp4_crf")
        )

        # Calculate relative path from OUTPUTS_ROOT
        outputs_mount = Path.home() / "Documents/moe-3d-reconstruction/outputs"
        rel_dir = Path(metadata_dir).relative_to(outputs_mount)

        result = {
            "video_filename": str(rel_dir / os.path.basename(output_filename)),
            "start_frame": str(rel_dir / "start_frame.png"),
            "end_frame": str(rel_dir / "end_frame.png"),
        }

        # Unload LoRAs
        if selected_loras:
            print("Unloading LoRAs")
            generator.unload_loras()
            torch.cuda.empty_cache()

    except Exception as e:
        traceback.print_exc()

        if generator and selected_loras:
            print("Unloading LoRAs after error")
            generator.unload_loras()
            torch.cuda.empty_cache()

        if not high_vram:
            unload_complete_models(
                text_encoder,
                text_encoder_2,
                image_encoder,
                vae,
                generator.transformer if generator else None,
            )

        raise

    finally:
        # Cleanup intermediate videos
        if settings.get("clean_up_videos"):
            CleanupManager.cleanup_intermediate_videos(output_dir, job_id)

        # Verify LoRA state
        LoRAManager.verify_lora_state(generator)

    return result
