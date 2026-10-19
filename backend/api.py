"""Local-only FastAPI upload/analysis API. No persistent video storage."""
import os
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from .analysis import VideoError, analyze, inspect

MAX_BYTES = 250 * 1024 * 1024
ALLOWED = {'.mp4', '.mov', '.avi', '.m4v'}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title='RoboGyn-AI Local API', version='0.2.0', lifespan=lifespan,
              description='Research-only local video analysis; no automatic skill grading.')

@app.get('/health')
def health():
    return {'status': 'ok', 'research_only': True}

async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, 'Unsupported extension. Use MP4, MOV, AVI, or M4V')
    tmp = tempfile.NamedTemporaryFile(prefix='robogyn_', suffix=suffix, delete=False)
    path = Path(tmp.name)
    total = 0
    try:
        with tmp:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, 'Video exceeds 250 MB limit')
                tmp.write(chunk)
        if total == 0:
            raise HTTPException(400, 'Empty upload')
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

@app.post('/api/v1/video/inspect')
async def inspect_endpoint(file: UploadFile = File(...)):
    path = await save_upload(file)
    try:
        return {'metadata': inspect(path), 'stored': False}
    except VideoError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        path.unlink(missing_ok=True)

@app.post('/api/v1/suturing/analyze')
async def analyze_endpoint(file: UploadFile = File(...),
                           start_s: float = Form(...), end_s: float = Form(...),
                           sample_fps: float = Form(2.0), crop_top_fraction: float = Form(0.1),
                           experience_level: str = Form('beginner')):
    if experience_level != 'beginner':
        raise HTTPException(422, 'This prototype currently supports beginner assessments only')
    path = await save_upload(file)
    try:
        return analyze(path, start_s, end_s, sample_fps, crop_top_fraction)
    except VideoError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        path.unlink(missing_ok=True)
