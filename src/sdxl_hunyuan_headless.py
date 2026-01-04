#!/usr/bin/env python3
import os
from pathlib import Path
from types import SimpleNamespace

from .framepack_core.queue_runner import QueueRunner

ROOT = Path(__file__).resolve().parent.parent  # .../moe-3d-reconstruction

OUTPUT_DIR = ROOT / "outputs"
DB_PATH = ROOT / "job_queue.sqlite"
os.environ["HF_HOME"] = os.path.abspath("./hf_download")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = SimpleNamespace(
    share=False,
    server="0.0.0.0",
    port=None,
    inbrowser=False,
    lora=None,
    offline=False,
    gpu_memory_preservation=14,
)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    runner = QueueRunner(DB_PATH, OUTPUT_DIR, args)
    runner.run_forever()


if __name__ == "__main__":
    main()
