import os
import time
from pathlib import Path

import torch
from types import SimpleNamespace
from transformers import SiglipImageProcessor, SiglipVisionModel
from .generator import HunyuanVideoGenerator
from .settings import Settings
from .diffusers_helper.memory import gpu, get_cuda_free_memory_gb

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HUNYUAN_MODEL = "hunyuanvideo-community/HunyuanVideo"
OUTPUT_DIR = Path("outputs/hunyuan_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_VISION_MODEL = "lllyasviel/flux_redux_bfl"
PROMPT = (
    "a cinematic wide shot of a futuristic drone flying through a neon-lit city, "
    "dynamic motion, realistic lighting, ultra-detailed"
)
os.environ["HF_HOME"] = os.path.abspath(
    "./hf_download"
)  # to load model downloaded offline
NEGATIVE_PROMPT = "low quality, blurry, jitter, inconsistent motion, deformed geometry"

SEED = 42

# -----------------------------------------------------------------------------
# Settings builder
# -----------------------------------------------------------------------------


def create_settings(high_vram: bool) -> Settings:
    settings = Settings()
    settings.set("model_path", HUNYUAN_MODEL)
    settings.set("low_vram_mode", not high_vram)
    settings.set("gpu_memory_preservation", 4)
    settings.set("use_fake_vae", False)

    settings.set("width", 320)
    settings.set("height", 320)
    settings.set("num_frames", 8)
    settings.set("num_inference_steps", 10)
    settings.set("guidance_scale", 7.5)

    settings.set("dtype", "float16")
    settings.set("device", "cuda")
    settings.set("attention_backend", "sage")
    settings.set("cpu_offload", True)

    return settings


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    print(f"[INFO] Free VRAM: {free_mem_gb:.1f} GB")
    print(f"[INFO] High VRAM mode: {high_vram}")

    settings = create_settings(high_vram)

    generator = HunyuanVideoGenerator(
        settings=settings,
        high_vram=high_vram,
        gpu_device=gpu,
    )

    print("[INFO] Loading Hunyuan models...")
    generator.load_models(HUNYUAN_MODEL)
    generator.configure_models()
    generator.apply_memory_management()

    start_time = time.time()
    feature_extractor = SiglipImageProcessor.from_pretrained(
        CLIP_VISION_MODEL, subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        CLIP_VISION_MODEL, subfolder="image_encoder", torch_dtype=torch.float16
    ).cpu()

    print("[INFO] Generating video...")
    result = generator.generate_video(
        None,
        None,
        PROMPT,
        NEGATIVE_PROMPT,
        SEED,
        feature_extractor,
        image_encoder,
        total_second_length=4.0,
        latent_window_size=8,
        steps=10,
        cfg=1.0,
        gs=0.0,
        rs=0.0,
        latent_type="black",
        has_input_image=False,
        use_magcache=True,
        magcache_threshold=0.15,
        magcache_max_consecutive_skips=10,
        magcache_retention_ratio=0.9,
        selected_loras=[],
        blend_sections=0,
        output_dir=str(OUTPUT_DIR),
        metadata_dir=str(OUTPUT_DIR),
        resolutionW=320,
        resolutionH=320,
        save_metadata_checked=True,
        job_stream=None,
    )

    elapsed = time.time() - start_time
    print(f"[DONE] Generation finished in {elapsed:.2f}s")
    print(f"[OUTPUT] Saved to: {OUTPUT_DIR.resolve()}")

    return result


if __name__ == "__main__":
    main()
