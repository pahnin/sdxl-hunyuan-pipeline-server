"""
Metadata utilities for FramePack Studio.
Provides functions for generating, saving, and managing metadata for video generation jobs.
"""

import os
import json
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from ..version import APP_VERSION


# =============================================================================
# COLOR MAPPING
# =============================================================================
def get_placeholder_color(model_type: str) -> tuple:
    """
    Get the placeholder image color for a specific model type.

    Args:
        model_type: The model type string

    Returns:
        RGB tuple for the placeholder image color
    """
    color_map = {
        "Original": (0, 0, 0),
        "F1": (0, 0, 128),
        "Video": (0, 128, 0),
        "XY Plot": (128, 128, 0),
        "F1 with Endframe": (0, 128, 128),
        "Original with Endframe": (128, 0, 128),
    }
    return color_map.get(model_type, (0, 0, 0))


# =============================================================================
# IMAGE NORMALIZATION
# =============================================================================
def normalize_image_to_uint8(image_np: np.ndarray) -> np.ndarray:
    """
    Normalize image array to uint8 format [0, 255].

    Args:
        image_np: Input image array (any dtype, any range)

    Returns:
        Normalized uint8 image array
    """
    if image_np.dtype == np.uint8:
        return image_np

    # Handle float images in range [-1, 1]
    if image_np.dtype in [np.float32, np.float64]:
        if image_np.max() <= 1.0 and image_np.min() >= -1.0:
            return ((image_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        # Handle float images in range [0, 1]
        elif image_np.max() <= 1.0 and image_np.min() >= 0.0:
            return (image_np * 255.0).clip(0, 255).astype(np.uint8)

    # Default: clip and convert
    return image_np.clip(0, 255).astype(np.uint8)


# =============================================================================
# METADATA CREATION
# =============================================================================
def create_png_metadata(job_params: dict) -> PngInfo:
    """
    Create PNG metadata from job parameters.

    Args:
        job_params: Dictionary of job parameters

    Returns:
        PngInfo object with embedded metadata
    """
    metadata = PngInfo()
    metadata.add_text("prompt", job_params.get("prompt_text", ""))
    metadata.add_text("seed", str(job_params.get("seed", 0)))
    metadata.add_text("model_type", job_params.get("model_type", "Unknown"))

    # Add XY Plot specific metadata
    if job_params.get("model_type") == "XY Plot":
        metadata.add_text("x_param", job_params.get("x_param", ""))
        metadata.add_text("y_param", job_params.get("y_param", ""))

    return metadata


def determine_end_frame_used(end_frame_image) -> bool:
    """
    Safely determine if end frame was used, handling NumPy array boolean ambiguity.

    Args:
        end_frame_image: End frame image (can be None, ndarray, or other)

    Returns:
        Boolean indicating if end frame was used
    """
    if end_frame_image is None:
        return False

    if isinstance(end_frame_image, np.ndarray):
        return bool(end_frame_image.any())

    return True


def extract_lora_metadata(job_params: dict) -> dict:
    """
    Extract LoRA information from job parameters.

    Args:
        job_params: Dictionary of job parameters

    Returns:
        Dictionary mapping LoRA names to their weights
    """
    selected_loras = job_params.get("selected_loras", [])
    lora_values = job_params.get("lora_values", [])
    lora_loaded_names = job_params.get("lora_loaded_names", [])

    if not isinstance(selected_loras, list) or len(selected_loras) == 0:
        return {}

    lora_data = {}
    for lora_name in selected_loras:
        try:
            idx = lora_loaded_names.index(lora_name)
            weight = lora_values[idx] if lora_values and idx < len(lora_values) else 1.0

            # Handle different weight types
            if isinstance(weight, np.ndarray):
                weight_value = (
                    float(weight.item()) if weight.size == 1 else float(weight.mean())
                )
            elif isinstance(weight, list):
                weight_value = float(weight[0]) if weight else 1.0
            else:
                weight_value = float(weight) if weight is not None else 1.0

            lora_data[lora_name] = weight_value
        except (ValueError, IndexError, TypeError):
            lora_data[lora_name] = 1.0
        except Exception:
            traceback.print_exc()
            lora_data[lora_name] = 1.0

    return lora_data


def create_metadata(
    job_params: dict, job_id: str, settings: dict, save_placeholder: bool = False
) -> dict:
    """
    Create comprehensive metadata dictionary for a job.

    Args:
        job_params: Dictionary of job parameters
        job_id: The job ID
        settings: Dictionary of settings
        save_placeholder: Whether to save placeholder image (default: False)

    Returns:
        Comprehensive metadata dictionary
    """
    if not settings.get("save_metadata"):
        return None

    # Ensure directories exist
    metadata_dir_path = settings.get("metadata_dir")
    output_dir_path = settings.get("output_dir")
    os.makedirs(metadata_dir_path, exist_ok=True)
    os.makedirs(output_dir_path, exist_ok=True)

    # Get model type and resolution
    model_type = job_params.get("model_type", "Original")
    height = job_params.get("height") or job_params.get("resolutionH", 640)
    width = job_params.get("width") or job_params.get("resolutionW", 640)

    # Create comprehensive metadata dictionary
    metadata_dict = {
        # Version information
        "app_version": APP_VERSION,
        # Common parameters
        "prompt": job_params.get("prompt_text", ""),
        "negative_prompt": job_params.get("n_prompt", ""),
        "seed": job_params.get("seed", 0),
        "steps": job_params.get("steps", 25),
        "cfg": job_params.get("cfg", 1.0),
        "gs": job_params.get("gs", 10.0),
        "rs": job_params.get("rs", 0.0),
        "latent_type": job_params.get("latent_type", "Black"),
        "timestamp": time.time(),
        "resolutionW": width,
        "resolutionH": height,
        "model_type": model_type,
        "generation_type": job_params.get("generation_type", model_type),
        "has_input_image": job_params.get("has_input_image", False),
        "input_image_path": job_params.get("input_image_path", None),
        # Video-related parameters
        "total_second_length": job_params.get("total_second_length", 6),
        "blend_sections": job_params.get("blend_sections", 4),
        "latent_window_size": job_params.get("latent_window_size", 9),
        "num_cleaned_frames": job_params.get("num_cleaned_frames", 5),
        # Endframe-related parameters
        "end_frame_strength": job_params.get("end_frame_strength", None),
        "end_frame_image_path": job_params.get("end_frame_image_path", None),
        "end_frame_used": str(
            determine_end_frame_used(job_params.get("end_frame_image"))
        ),
        # Video input parameters
        "input_video": (
            os.path.basename(job_params.get("input_image", ""))
            if job_params.get("input_image") is not None and model_type == "Video"
            else None
        ),
        "video_path": job_params.get("input_image") if model_type == "Video" else None,
        # XY Plot parameters
        "x_param": job_params.get("x_param", None),
        "y_param": job_params.get("y_param", None),
        "x_values": job_params.get("x_values", None),
        "y_values": job_params.get("y_values", None),
        # Combine with source video
        "combine_with_source": job_params.get("combine_with_source", False),
        # Cache parameters
        "use_teacache": job_params.get("use_teacache", False),
        "teacache_num_steps": job_params.get("teacache_num_steps", 0),
        "teacache_rel_l1_thresh": job_params.get("teacache_rel_l1_thresh", 0.0),
        "use_magcache": job_params.get("use_magcache", False),
        "magcache_threshold": job_params.get("magcache_threshold", 0.1),
        "magcache_max_consecutive_skips": job_params.get(
            "magcache_max_consecutive_skips", 2
        ),
        "magcache_retention_ratio": job_params.get("magcache_retention_ratio", 0.25),
        # LoRA information
        "loras": extract_lora_metadata(job_params),
    }

    # Create and save placeholder image if requested
    if save_placeholder:
        _save_placeholder_image(
            metadata_dir_path, job_id, model_type, width, height, job_params
        )

    return metadata_dict


def _save_placeholder_image(
    metadata_dir_path: str,
    job_id: str,
    model_type: str,
    width: int,
    height: int,
    job_params: dict,
):
    """Save placeholder image with XY plot text if applicable."""
    placeholder_color = get_placeholder_color(model_type)
    placeholder_img = Image.new("RGB", (width, height), placeholder_color)

    # Add XY plot parameters to image
    if model_type == "XY Plot":
        x_param = job_params.get("x_param", "")
        y_param = job_params.get("y_param", "")
        x_values = job_params.get("x_values", [])
        y_values = job_params.get("y_values", [])

        draw = ImageDraw.Draw(placeholder_img)
        try:
            font = ImageFont.truetype("Arial", 20)
        except:
            font = ImageFont.load_default()

        text = f"X: {x_param} - {x_values}\nY: {y_param} - {y_values}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    # Create PNG metadata and save
    png_metadata = create_png_metadata(job_params)
    placeholder_path = os.path.join(metadata_dir_path, f"{job_id}.png")

    try:
        placeholder_img.save(placeholder_path, pnginfo=png_metadata)
    except Exception:
        traceback.print_exc()


# =============================================================================
# IMAGE SAVING
# =============================================================================
def save_job_start_image(job_params: dict, job_id: str, settings: dict) -> bool:
    """
    Save the job's starting input image with comprehensive metadata.
    This should be called early in job processing.

    Args:
        job_params: Dictionary of job parameters
        job_id: The job ID
        settings: Dictionary of settings

    Returns:
        Boolean indicating success or failure
    """
    output_dir_path = job_params.get("output_dir") or settings.get("output_dir")
    metadata_dir_path = job_params.get("metadata_dir") or settings.get("metadata_dir")

    if not output_dir_path:
        print(f"[JOB_START_IMG_ERROR] No output directory found")
        return False

    # Ensure directories exist
    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(metadata_dir_path, exist_ok=True)

    # Create metadata
    metadata_dict = create_metadata(job_params, job_id, settings)

    # Save JSON metadata
    json_metadata_path = os.path.join(metadata_dir_path, f"{job_id}.json")
    try:
        with open(json_metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
    except Exception:
        traceback.print_exc()

    # Save input image if available
    input_image_np = job_params.get("input_image")
    if input_image_np is not None and isinstance(input_image_np, np.ndarray):
        return _save_image_with_metadata(
            image_np=input_image_np,
            output_path=os.path.join(output_dir_path, f"{job_id}.png"),
            metadata_dict=metadata_dict,
            job_params=job_params,
        )

    return False


def save_last_video_frame(
    job_params: dict, job_id: str, settings: dict, last_frame_np: np.ndarray
) -> bool:
    """
    Save the last frame of the input video with metadata.

    Args:
        job_params: Dictionary of job parameters
        job_id: The job ID
        settings: Dictionary of settings
        last_frame_np: Last frame as NumPy array

    Returns:
        Boolean indicating success or failure
    """
    output_dir_path = job_params.get("output_dir") or settings.get("output_dir")

    if not output_dir_path:
        print(f"[SAVE_LAST_FRAME_ERROR] No output directory found")
        return False

    os.makedirs(output_dir_path, exist_ok=True)

    metadata_dict = create_metadata(job_params, job_id, settings)

    if last_frame_np is not None and isinstance(last_frame_np, np.ndarray):
        output_path = os.path.join(output_dir_path, f"{job_id}.png")
        success = _save_image_with_metadata(
            image_np=last_frame_np,
            output_path=output_path,
            metadata_dict=metadata_dict,
            job_params=job_params,
        )
        if success:
            print(f"Saved last video frame for job {job_id} to {output_path}")
        return success

    return False


def _save_image_with_metadata(
    image_np: np.ndarray, output_path: str, metadata_dict: dict, job_params: dict
) -> bool:
    """
    Save image array as PNG with embedded metadata.

    Args:
        image_np: Image array to save
        output_path: Full path to save image
        metadata_dict: Metadata dictionary
        job_params: Job parameters for PNG metadata

    Returns:
        Boolean indicating success or failure
    """
    try:
        # Create PNG metadata
        png_metadata = PngInfo()
        for key, value in metadata_dict.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                png_metadata.add_text(key, str(value))

        # Normalize image to uint8
        image_uint8 = normalize_image_to_uint8(image_np)

        # Save image
        image_pil = Image.fromarray(image_uint8)
        image_pil.save(output_path, pnginfo=png_metadata)
        return True

    except Exception:
        traceback.print_exc()
        return False
