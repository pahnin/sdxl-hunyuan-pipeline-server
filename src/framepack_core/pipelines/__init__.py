"""
Pipeline module for FramePack Studio.
This module provides pipeline classes for different generation types.
"""

from .base_pipeline import BasePipeline
from .original_pipeline import OriginalPipeline
from .original_with_endframe_pipeline import OriginalWithEndframePipeline


def create_pipeline(
    model_type, settings, output_dir, metadata_dir, input_files_dir, high_vram
):
    """
    Create a pipeline instance for the specified model type.

    Args:
        model_type: The type of model to create a pipeline for
        settings: Dictionary of settings for the pipeline

    Returns:
        A pipeline instance for the specified model type
    """
    pipeline_settings = {
        "output_dir": output_dir,
        "metadata_dir": metadata_dir,
        "input_files_dir": input_files_dir,
        "save_metadata": settings.get("save_metadata", True),
        "gpu_memory_preservation": settings.get("gpu_memory_preservation", 6),
        "mp4_crf": settings.get("mp4_crf", 16),
        "clean_up_videos": settings.get("clean_up_videos", True),
        "gradio_temp_dir": settings.get("gradio_temp_dir", "./gradio_temp"),
        "high_vram": high_vram,
    }
    if model_type == "Original":
        return OriginalPipeline(pipeline_settings)
    elif model_type == "Original with Endframe":
        return OriginalWithEndframePipeline(pipeline_settings)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "BasePipeline",
    "OriginalPipeline",
    "OriginalWithEndframePipeline",
    "create_pipeline",
]
