# =============================================================================
# SDXL TURBO GENERATOR (STATIC STATE FRAMES)
# =============================================================================
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import torch
from .framepack_core.diffusers_helper.memory import gpu
import numpy as np


class SDXLFrameGenerator:
    # Constants for generation parameters
    DEFAULT_NEGATIVE_PROMPT = (
        "motion blur, animation, cartoon, illustration, CGI, "
        "distorted geometry, soft edges, bad anatomy, unrealistic"
    )

    START_FRAME_PROMPT_SUFFIX = (
        ", static scene, initial position, objects clearly visible, "
        "sharp focus, high detail, clean composition"
    )
    START_FRAME_NEGATIVE_PROMPT = (
        "motion blur, movement, dynamic action, blurry, low quality, "
        "artifacts, bad anatomy, impossible physics"
    )

    END_FRAME_PROMPT_SUFFIX = (
        ", action completed, final position, motion finished, "
        "sharp focus, high detail, dynamic composition"
    )
    END_FRAME_NEGATIVE_PROMPT = (
        "static, frozen motion, blurry, low quality, artifacts, "
        "bad anatomy, impossible physics"
    )

    DEFAULT_INFERENCE_STEPS = 4
    SPECIALIZED_INFERENCE_STEPS = 2
    GUIDANCE_SCALE = 1.0

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipe = None

    def load(self):
        if self.pipe is None:
            print("[SDXL] Loading SDXL Turbo…")
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.set_progress_bar_config(disable=True)
        return self.pipe

    def unload(self):
        if self.pipe is not None:
            self.pipe.to("cpu")
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("[SDXL] Unloaded")

    @torch.no_grad()
    def generate(self, prompt: str, width: int, height: int, seed: int):
        """Generate a standard frame with default settings."""
        return self._generate_with_prompt(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=self.DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=self.DEFAULT_INFERENCE_STEPS,
        )

    @torch.no_grad()
    def generate_start_frame(self, prompt: str, width: int, height: int, seed: int):
        """Generate a start frame optimized for static initial state."""
        start_prompt = f"{prompt}{self.START_FRAME_PROMPT_SUFFIX}"
        return self._generate_with_prompt(
            prompt=start_prompt,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=self.START_FRAME_NEGATIVE_PROMPT,
            num_inference_steps=self.SPECIALIZED_INFERENCE_STEPS,
        )

    @torch.no_grad()
    def generate_end_frame(self, prompt: str, width: int, height: int, seed: int):
        """Generate an end frame optimized for final state with motion."""
        end_prompt = f"{prompt}{self.END_FRAME_PROMPT_SUFFIX}"
        return self._generate_with_prompt(
            prompt=end_prompt,
            width=width,
            height=height,
            seed=seed,
            negative_prompt=self.END_FRAME_NEGATIVE_PROMPT,
            num_inference_steps=self.SPECIALIZED_INFERENCE_STEPS,
        )

    @torch.no_grad()
    def _generate_with_prompt(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        negative_prompt: str,
        num_inference_steps: int,
    ):
        """Core generation method used by all public generate methods."""
        pipe = self.load()
        gen = torch.Generator(device=gpu).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.GUIDANCE_SCALE,
            width=width,
            height=height,
            generator=gen,
        ).images[0]
        return np.array(image, dtype=np.uint8)
