import signal
import sys
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace
import sqlite3

from .job_queue import JobQueue


def worker_process(job_id, payload, job_queue, db_path, output_dir, args):
    """Run generation in a separate process."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    from ..sdxl_hunyuan_video_pipeline import SdxlHunyuanVideoPipeline
    from .cancel_stream import CancelStream

    # from .queue_runner import update_job_outputs  # adjust import to where you put it

    try:
        generator = SdxlHunyuanVideoPipeline(output_dir, args)
        job_stream = CancelStream(db_path, job_id)

        result = generator.generate_video(
            obj=payload["obj"],
            index=payload.get("index", job_id),
            seed=payload.get("seed", 2000 + job_id),
            resolution=payload.get("resolution", 480),
            duration=payload.get("duration", 5),
            latent_window=payload.get("latent_window", 8),
            num_steps=payload.get("num_steps", 30),
            cfg_scale=payload.get("cfg_scale", 1.0),
            job_stream=job_stream,
        )

        print(f"result from generator {result}")
        # result is expected to be a dict with filenames from inner worker
        video_filename = result.get("video_filename")
        preview_image = result.get("preview_image")

        if video_filename:
            job_queue.update_job_outputs(
                job_id,
                Path(output_dir),
                result.get("video_filename", ""),
                result.get("start_frame", ""),
                result.get("end_frame", ""),
            )

        print(f"[Worker] Job {job_id} completed: {result}")
        return 0  # Success

    except KeyboardInterrupt:
        print(f"[Worker] Job {job_id} interrupted")
        return 1  # Interrupted
    except Exception as e:
        print(f"[Worker] Job {job_id} failed: {e}")
        return 2  # Failed


class QueueRunner:
    def __init__(self, db_path: Path, output_dir: Path, args: SimpleNamespace):
        self.queue = JobQueue(db_path)
        self.output_dir = output_dir
        self.args = args
        self.current_process = None
        self.shutdown_flag = False

        # Only register handlers in main process
        if multiprocessing.current_process().name == "MainProcess":
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals in main process only."""
        if self.shutdown_flag:
            return  # Already shutting down

        print(f"\n[SHUTDOWN] Signal {signum} received, terminating...")
        self.shutdown_flag = True

        # Only terminate if process exists and is alive
        if self.current_process and self.current_process.is_alive():
            print(f"[SHUTDOWN] Terminating process {self.current_process.pid}...")
            try:
                self.current_process.terminate()
                # Wait up to 10 seconds for graceful shutdown
                self.current_process.join(timeout=10)

                if self.current_process.is_alive():
                    print("[SHUTDOWN] Process didn't terminate, killing...")
                    self.current_process.kill()
                    self.current_process.join(timeout=5)
            except Exception as e:
                print(f"[SHUTDOWN] Error terminating process: {e}")

        # Exit immediately
        sys.exit(0)

    def run_forever(self, poll_interval: float = 2.0):
        """Process jobs with process-based isolation."""
        print(f"[QueueRunner] Started in process {os.getpid()}")

        while not self.shutdown_flag:
            try:
                job = self.queue.fetch_next_pending()

                if not job:
                    time.sleep(poll_interval)
                    continue

                # Check if cancelled before processing
                if self.queue.get_status(job.id) == "cancelled":
                    print(f"[SKIP] Job {job.id} was cancelled, skipping...")
                    continue

                # Run in separate process
                self.current_process = multiprocessing.Process(
                    target=worker_process,
                    args=(
                        job.id,
                        job.payload,
                        self.queue,
                        self.queue.db_path,
                        self.output_dir,
                        self.args,
                    ),
                    daemon=True,  # Kill if parent exits
                )
                self.current_process.start()
                self.current_process.join()

                # Check exit code
                if self.current_process.exitcode == 0:
                    status = self.queue.get_status(job.id)
                    if status != "cancelled":
                        self.queue.mark_done(job.id)
                elif self.current_process.exitcode == 1:
                    if self.queue.get_status(job.id) != "cancelled":
                        self.queue.mark_failed(job.id, "Interrupted by user")
                else:
                    if self.queue.get_status(job.id) != "cancelled":
                        self.queue.mark_failed(job.id, "Process crashed or failed")

                self.current_process = None

            except KeyboardInterrupt:
                print("\n[QueueRunner] Interrupted, exiting...")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] Queue runner error: {e}")
                time.sleep(5)

        print("[SHUTDOWN] Queue runner stopped gracefully.")
