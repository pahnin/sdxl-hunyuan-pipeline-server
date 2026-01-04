import os


# =============================================================================
# CLEANUP MANAGER
# =============================================================================
class CleanupManager:
    """Handles cleanup of intermediate video files."""

    @staticmethod
    def cleanup_intermediate_videos(output_dir: str, job_id: str):
        """Remove intermediate video files, keeping only the final one."""
        try:
            video_files = [
                f
                for f in os.listdir(output_dir)
                if f.startswith(f"{job_id}_") and f.endswith(".mp4")
            ]
            print(f"Video files found for cleanup: {video_files}")

            if not video_files:
                return

            def get_frame_count(filename):
                try:
                    return int(filename.replace(f"{job_id}_", "").replace(".mp4", ""))
                except Exception:
                    return -1

            video_files_sorted = sorted(video_files, key=get_frame_count)
            print(f"Sorted video files: {video_files_sorted}")

            # Delete all but the last (final) video
            for vf in video_files_sorted[:-1]:
                full_path = os.path.join(output_dir, vf)
                try:
                    os.remove(full_path)
                    print(f"Deleted intermediate video: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")

        except Exception as e:
            print(f"Error during video cleanup: {e}")
