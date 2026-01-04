import time
from datetime import timedelta


# =============================================================================
# PROGRESS TRACKER
# =============================================================================
class ProgressTracker:
    """Tracks generation progress and estimates ETA."""

    def __init__(self, total_steps: int, total_second_length: float, job_stream=None):
        self.total_steps = total_steps
        self.total_second_length = total_second_length
        self.step_durations = []
        self.last_step_time = time.time()
        self.job_stream = job_stream

    def create_callback(self, total_generated_latent_frames):
        def callback(d):
            now_time = time.time()

            if self.last_step_time is not None:
                step_delta = now_time - self.last_step_time
                if step_delta > 0:
                    self.step_durations.append(step_delta)
                    if len(self.step_durations) > 30:
                        self.step_durations.pop(0)

            self.last_step_time = now_time

            # positions
            current_pos = max(0, (total_generated_latent_frames * 4 - 3) / 30.0)
            original_pos = max(0, self.total_second_length - current_pos)

            # optional ETA logging
            if self.step_durations:
                avg = sum(self.step_durations) / len(self.step_durations)
                remaining_steps = max(0, self.total_steps - d.get("step", 0))
                eta_seconds = remaining_steps * avg
                # print(f"ETA: {self.format_eta(eta_seconds)}")

            # cancellation hook
            if self.job_stream is not None and self.job_stream.is_cancelled():
                raise RuntimeError("JobCancelled")

        return callback

    @staticmethod
    def format_eta(seconds: float) -> str:
        """Format ETA in human-readable format."""
        try:
            return str(timedelta(seconds=int(seconds)))
        except Exception:
            return "--:--"
