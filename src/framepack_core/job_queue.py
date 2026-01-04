import json
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Job:
    id: int
    status: str
    payload: dict
    attempt_count: int


class JobQueue:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                ON jobs(status, created_at);
            """
            )

    def enqueue(self, payload: dict) -> int:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO jobs(status, payload_json, created_at, updated_at) "
                "VALUES(?, ?, ?, ?)",
                ("pending", json.dumps(payload), now, now),
            )
            conn.commit()
            return cur.lastrowid

    def fetch_next_pending(self) -> Job | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM jobs WHERE status = 'pending' "
                "ORDER BY created_at LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                return None

            conn.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), row["id"]),
            )
            conn.commit()

            return Job(
                id=row["id"],
                status="running",
                payload=json.loads(row["payload_json"]),
                attempt_count=row["attempt_count"],
            )

    def mark_done(self, job_id: int):
        self._update_status(job_id, "done")

    def mark_failed(self, job_id: int, error: str):
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', attempt_count = attempt_count + 1, "
                "last_error = ?, updated_at = ? WHERE id = ?",
                (error, now, job_id),
            )
            conn.commit()

    def mark_cancelled(self, job_id: int):
        self._update_status(job_id, "cancelled")

    def _update_status(self, job_id: int, status: str):
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, job_id),
            )
            conn.commit()

    def get_status(self, job_id: int) -> str | None:
        """Check if a job is cancelled before processing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return row["status"] if row else None

    def update_job_outputs(
        self,
        job_id: int,
        output_dir: Path,  # This is metadata_dir from worker
        video_filename: str,  # Already relative
        start_frame: str,  # Already relative
        end_frame: str,  # Already relative
    ):
        """Update job with output file information."""
        from datetime import datetime

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE jobs
                   SET status = 'done',
                       output_dir = ?,
                       video_filename = ?,
                       start_frame = ?,
                       end_frame = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (
                    str(
                        output_dir
                    ),  # Absolute: /home/phanindra/.../2026-01-03/13_30_27_500_018_Indian_modern_dancer
                    video_filename,  # Relative: 2026-01-03/13_30_27_500_018_Indian_modern_dancer/260103_133059_617_6639_25.mp4
                    start_frame,  # Relative: 2026-01-03/13_30_27_500_018_Indian_modern_dancer/start_frame.png
                    end_frame,  # Relative: 2026-01-03/13_30_27_500_018_Indian_modern_dancer/end_frame.png
                    datetime.now().isoformat(),
                    job_id,
                ),
            )
            conn.commit()
