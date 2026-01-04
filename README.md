# SDXL Hunyuan Pipeline Server

Generates **Framepack videos from images** via **SDXL → Hunyuan → Framepack pipeline** with job queue management.

This repository contains the **headless SDXL image-to-Framepack video queue server** with **FastAPI job management**.

## Quick Start

### Server (Job Queue Management)
```
python src/server.py
```
Runs the FastAPI server for job submission and queue management.

### Queue Processor (Headless Worker)
```
python -m src.sdxl_hunyuan_headless
```
Processes jobs from the queue (SDXL → Hunyuan → Framepack pipeline).

## Android Client
The companion mobile app is available at:  
[sdxl-hunyuan-pipeline-android-client](https://github.com/pahnin/sdxl-hunyuan-pipeline-android-client)

**Note:** Both processes must run simultaneously for full pipeline operation. The Android client submits jobs to `server.py`, which the `sdxl_hunyuan_headless` worker processes.[1]

[1](https://github.com/pahnin/sdxl-hunyuan-pipeline-server)
