import os
import json
import time
import psutil
import torch
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import numpy as np
from transformers import SiglipImageProcessor, SiglipVisionModel
from .framepack_core.diffusers_helper.memory import gpu, get_cuda_free_memory_gb
from .framepack_core.llm_enhancer import enhance_prompt
from .sdxl_frame_generator import SDXLFrameGenerator
from .framepack_core.generator import HunyuanVideoGenerator
from .framepack_core.settings import Settings

# Constants
HUNYUAN_MODEL = "hunyuanvideo-community/HunyuanVideo"
CLIP_VISION_MODEL = "lllyasviel/flux_redux_bfl"
DEFAULT_SDXL_TURBO_PATH = (
    Path.home() / "Documents/composition-pipeline/models/sdxl-turbo"
)
HIGH_VRAM_THRESHOLD_GB = 60
END_FRAME_SEED_OFFSET = 999


# Prompt builders
def build_state_prompt(obj, state_desc):
    """Build prompt for a specific object state (start/end)."""
    return (
        f"wide shot, single {obj['name']}, {state_desc}, {obj['view']}, "
        "studio product photography, white cyclorama, "
        "sharp edges, realistic materials, true-to-scale"
    )


def build_hunyuan_prompt(obj):
    """Build and enhance prompt for HunyuanVideo generation."""
    base = f"single {obj['name']}, {obj['view']}, studio environment, {obj['motion']}"
    return enhance_prompt(base)


class SdxlHunyuanVideoPipeline:
    """Orchestrates video generation using SDXL and HunyuanVideo models."""

    def __init__(
        self,
        output_dir: Path,
        args: SimpleNamespace,
        sdxl_model_path: Path = DEFAULT_SDXL_TURBO_PATH,
    ):
        self.output_dir = output_dir
        self.sdxl = SDXLFrameGenerator(sdxl_model_path)
        self.process = psutil.Process(os.getpid())

        # VRAM detection
        free_mem_gb = get_cuda_free_memory_gb(gpu)
        self.high_vram = free_mem_gb > HIGH_VRAM_THRESHOLD_GB

        # Initialize dedicated generator
        settings = self._create_settings(args)
        self.hunyuan_gen = HunyuanVideoGenerator(settings, self.high_vram, gpu)

        # Load CLIP vision models
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            CLIP_VISION_MODEL, subfolder="feature_extractor"
        )
        self.image_encoder = SiglipVisionModel.from_pretrained(
            CLIP_VISION_MODEL, subfolder="image_encoder", torch_dtype=torch.float16
        ).cpu()

    def _create_settings(self, args: SimpleNamespace) -> Settings:
        """Initialize settings based on VRAM availability and arguments."""
        settings = Settings()
        settings.set("model_path", HUNYUAN_MODEL)
        settings.set("gpu_memory_preservation", args.gpu_memory_preservation)
        settings.set("low_vram_mode", not self.high_vram)
        settings.set("use_fake_vae", False)
        settings.set("height", 320)
        settings.set("width", 320)
        settings.set("num_frames", 8)
        settings.set("num_inference_steps", 30)
        settings.set("guidance_scale", 7.5)
        settings.set("dtype", "float16")
        settings.set("device", "cuda")
        settings.set("attention_backend", "sage")
        settings.set("cpu_offload", True)
        return settings

    def generate_video(
        self,
        *,
        obj,
        index,
        seed,
        resolution,
        duration,
        latent_window,
        num_steps,
        cfg_scale,
        job_stream=None,
    ):
        """Generate video for a given object with specified parameters."""
        start_time = time.time()

        # Create output folder
        folder = self._create_output_folder(index, obj["name"])

        # Build prompts
        start_prompt = build_state_prompt(obj, obj["initial"])
        end_prompt = build_state_prompt(obj, obj["final"])
        hunyuan_prompt = build_hunyuan_prompt(obj)
        n_prompt = "pixelated, deformed, bad anatomy, incohesive, inconsistent"

        self._print_generation_info(
            obj["name"], start_prompt, end_prompt, hunyuan_prompt
        )

        # Generate and save frames
        start_img, end_img = self._generate_frames(obj, seed, resolution)
        self._save_frames(start_img, end_img, folder)

        # Load and configure HunyuanVideo models
        models = self.hunyuan_gen.load_models(HUNYUAN_MODEL)
        self.hunyuan_gen.configure_models()
        self.hunyuan_gen.apply_memory_management()

        # Prepare generation parameters
        generation_params = {
            "total_second_length": duration,
            "latent_window_size": latent_window,
            "steps": num_steps,
            "cfg": cfg_scale,
            "gs": 0.0,
            "rs": 0.0,
            "use_magcache": True,
            "magcache_threshold": 0.15,
            "magcache_max_consecutive_skips": 10,
            "magcache_retention_ratio": 0.9,
            "blend_sections": 0,
            "latent_type": "black",
            "selected_loras": [],
            "has_input_image": True,
            "job_stream": job_stream,
            "output_dir": str(folder),
            "metadata_dir": str(folder),
            "resolutionW": resolution,
            "resolutionH": resolution,
            "save_metadata_checked": True,
        }

        # Generate video
        result = self.hunyuan_gen.generate_video(
            start_img=start_img,
            end_img=end_img,
            prompt=hunyuan_prompt,
            n_prompt=n_prompt,
            seed=seed,
            feature_extractor=self.feature_extractor,
            image_encoder=self.image_encoder,
            **generation_params,
        )

        # Save metadata
        generation_time = round(time.time() - start_time, 2)
        self._save_metadata(
            folder,
            obj,
            seed,
            resolution,
            duration,
            latent_window,
            num_steps,
            cfg_scale,
            hunyuan_prompt,
            n_prompt,
            generation_time,
        )

        # Cleanup
        self.sdxl.unload()

        return result

    def _create_output_folder(self, index: int, obj_name: str) -> Path:
        """Create timestamped output folder."""
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H_%M_%S_%f")[:-3]
        sanitized_name = (
            obj_name.strip().replace(" ", "_").replace("\t", "").replace("\n", "")
        )
        folder = (
            self.output_dir
            / date
            / f"{timestamp}_{index:03d}_{sanitized_name.replace(' ', '_')}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _print_generation_info(
        self, obj_name: str, start_prompt: str, end_prompt: str, hunyuan_prompt: str
    ):
        """Print generation information to console."""
        print("\n" + "=" * 80)
        print(f"[OBJECT] {obj_name}")
        print("[SDXL START]", start_prompt)
        print("[SDXL END]  ", end_prompt)
        print("[HUNYUAN]   ", hunyuan_prompt)
        print("=" * 80)

    def _generate_frames(self, obj, seed, resolution):
        """Generate start and end frames using SDXL."""
        start_prompt = build_state_prompt(obj, obj["initial"])
        end_prompt = build_state_prompt(obj, obj["final"])

        start_img = self.sdxl.generate_start_frame(
            start_prompt, resolution, resolution, seed
        )
        end_img = self.sdxl.generate_end_frame(
            end_prompt, resolution, resolution, seed + END_FRAME_SEED_OFFSET
        )
        return start_img, end_img

    def _save_frames(self, start_img, end_img, folder: Path):
        """Save frames to disk."""
        Image.fromarray(start_img).save(folder / "start_frame.png")
        Image.fromarray(end_img).save(folder / "end_frame.png")

    def _save_metadata(
        self,
        folder: Path,
        obj: dict,
        seed: int,
        resolution: int,
        duration: float,
        latent_window: int,
        num_steps: int,
        cfg_scale: float,
        hunyuan_prompt: str,
        n_prompt: str,
        generation_time: float,
    ):
        """Save generation metadata to JSON file."""
        metadata = {
            "seed": seed,
            "object": obj,
            "resolution": resolution,
            "video_duration": duration,
            "latent_window": latent_window,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "output_path": str(folder),
            "generation_time_seconds": generation_time,
            "prompt": hunyuan_prompt,
            "negative_prompt": n_prompt,
        }

        metadata_path = folder / "generation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
