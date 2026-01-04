import sqlite3
import time


class CancelStream:
    def __init__(self, db_path, job_id: int, poll_interval: float = 0.5):
        self.db_path = db_path
        self.job_id = job_id
        self.poll_interval = poll_interval
        self._last_check = 0.0

    def is_cancelled(self) -> bool:
        now = time.time()
        if now - self._last_check < self.poll_interval:
            return False
        self._last_check = now
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("SELECT status FROM jobs WHERE id = ?", (self.job_id,))
            row = cur.fetchone()
            return row and row[0] == "cancelled"
        finally:
            conn.close()
