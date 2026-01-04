from fastapi import FastAPI, HTTPException, Depends, status, Query, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import sqlite3
import json
import time
from pathlib import Path
from fastapi.staticfiles import StaticFiles

DB_PATH = Path("job_queue.sqlite")


# Pydantic models (SINGLE DEFINITION ONLY)
class ObjectModel(BaseModel):
    name: str
    view: str
    initial: str
    final: str
    motion: str


class JobPayload(BaseModel):
    obj: ObjectModel
    seed: Optional[int] = Field(default=2000, description="Random seed for generation")
    resolution: Optional[int] = Field(default=480, ge=64, le=1024)
    duration: Optional[float] = Field(default=5.0, ge=1.0, le=60.0)
    latent_window: Optional[int] = Field(default=8, ge=1, le=32)
    num_steps: Optional[int] = Field(default=30, ge=1, le=100)
    cfg_scale: Optional[float] = Field(default=1.0, ge=0.1, le=20.0)


class JobResponse(BaseModel):
    id: int
    status: str
    payload: dict
    created_at: str
    updated_at: str
    attempt_count: int
    last_error: Optional[str] = None
    output_dir: Optional[str] = None
    video_url: Optional[str] = None
    start_frame_url: Optional[str] = None
    end_frame_url: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


def get_db():
    """Provide database connection with proper cleanup."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                output_dir TEXT,              -- absolute path for reference
                video_filename TEXT,          -- relative path: 2026-01-03/.../video.mp4
                start_frame TEXT,             -- relative path: 2026-01-03/.../start_frame.png
                end_frame TEXT                -- relative path: 2026-01-03/.../end_frame.png
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                ON jobs(status, created_at);

            CREATE INDEX IF NOT EXISTS idx_jobs_id_status
                ON jobs(id, status);
            """
        )
        conn.commit()


# FastAPI app
app = FastAPI(
    title="Video Generation Job Queue API",
    description="API for managing video generation jobs with SDXL and HunyuanVideo",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch all exceptions and return JSON error response."""
    return ErrorResponse(
        error="Internal Server Error",
        detail=str(exc),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


@app.post(
    "/jobs/",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
def create_job(payload: JobPayload, db: sqlite3.Connection = Depends(get_db)):
    """Create a new video generation job in the queue."""
    try:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload_dict = payload.model_dump()

        cur = db.execute(
            "INSERT INTO jobs(status, payload_json, created_at, updated_at) "
            "VALUES(?, ?, ?, ?)",
            ("pending", json.dumps(payload_dict), now, now),
        )
        db.commit()
        job_id = cur.lastrowid

        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Job created successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}",
        )


@app.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,
)
def cancel_job(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    """Cancel a job if it's still pending or running."""
    try:
        row = db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
            )

        current_status = row["status"]

        if current_status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job with status '{current_status}'",
            )

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            ("cancelled", now, job_id),
        )
        db.commit()

        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}",
        )


def make_urls(base_url: str, row: dict) -> tuple:
    """Generate URLs for job outputs."""
    video = row["video_filename"]
    start = row["start_frame"]
    end = row["end_frame"]

    if not video:
        return (None, None, None, None)

    base = f"{base_url}/outputs"
    video_url = f"{base}/{video}" if video else None
    start_url = f"{base}/{start}" if start else None
    end_url = f"{base}/{end}" if end else None

    return (row["output_dir"], video_url, start_url, end_url)


@app.get(
    "/jobs/",
    response_model=List[JobResponse],
)
def list_jobs(
    status_filter: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: sqlite3.Connection = Depends(get_db),
    request: Request = None,
):
    """List jobs with optional status filter and pagination."""
    try:
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        if status_filter and status_filter not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter. Must be one of: {', '.join(valid_statuses)}",
            )

        query = "SELECT * FROM jobs"
        params = []

        if status_filter:
            query += " WHERE status = ?"
            params.append(status_filter)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = db.execute(query, params).fetchall()
        base_url = str(request.base_url).rstrip("/")

        return [
            JobResponse(
                id=row["id"],
                status=row["status"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                attempt_count=row["attempt_count"],
                last_error=row["last_error"],
                output_dir=row["output_dir"],
                video_url=video_url,
                start_frame_url=start_url,
                end_frame_url=end_url,
            )
            for row in rows
            for _, video_url, start_url, end_url in [make_urls(base_url, row)]
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}",
        )


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
)
def get_job(
    job_id: int,
    db: sqlite3.Connection = Depends(get_db),
    request: Request = None,
):
    """Get detailed information about a specific job by ID."""
    try:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        base_url = str(request.base_url).rstrip("/")
        _, video_url, start_url, end_url = make_urls(base_url, row)

        return JobResponse(
            id=row["id"],
            status=row["status"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            attempt_count=row["attempt_count"],
            last_error=row["last_error"],
            output_dir=row["output_dir"],
            video_url=video_url,
            start_frame_url=start_url,
            end_frame_url=end_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job {job_id}: {str(e)}",
        )


@app.get("/health")
def health_check():
    """Check API health status."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1").fetchone()

        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "database": "connected",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        )


outputs_path = Path.home() / "Documents/moe-3d-reconstruction/outputs"
app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
