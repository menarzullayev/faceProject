"""
Enrollment Service API
FastAPI backend for face enrollment — detection, alignment, embedding, DB export.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import FacePipeline
from database import EmployeeDB

try:
    from dotenv import load_dotenv
    # Load environment variables from a .env file if it exists
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRFD_MODEL = os.getenv(
    "SCRFD_MODEL",
    os.path.join(BASE_DIR, "models", "scrfd_10g_bnkps.onnx")
)
ARCFACE_MODEL = os.getenv(
    "ARCFACE_MODEL",
    os.path.join(BASE_DIR, "models", "w600k_mbf.onnx")
)
CRFIQA_MODEL = os.getenv(
    "CRFIQA_MODEL",
    os.path.join(BASE_DIR, "models", "crfiqa_s_quality_opset11.onnx")
)
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "employee.db"))

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Enrollment Service",
    description="Enrolls employees, builds camera-compatible employee.db",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> FacePipeline:
    if not hasattr(app.state, "pipeline"):
        app.state.pipeline = FacePipeline(SCRFD_MODEL, ARCFACE_MODEL, CRFIQA_MODEL)
    return app.state.pipeline


def get_db() -> EmployeeDB:
    if not hasattr(app.state, "db"):
        app.state.db = EmployeeDB(DB_PATH)
    return app.state.db


@app.on_event("startup")
async def startup():
    print(f"[startup] Loading SCRFD: {SCRFD_MODEL}")
    print(f"[startup] Loading ArcFace: {ARCFACE_MODEL}")
    print(f"[startup] Loading CR-FIQA: {CRFIQA_MODEL}")
    get_pipeline()
    get_db()
    print(f"[startup] DB: {DB_PATH}")
    print(f"[startup] Stats: {get_db().stats()}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PersonOut(BaseModel):
    id: int
    name: str
    embed_count: int


class EnrollResult(BaseModel):
    person_id: int
    name: str
    photos_processed: int
    photos_failed: int
    embed_count: int
    failed_files: List[str]


class StatsOut(BaseModel):
    persons: int
    embeddings: int
    size_kb: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    min_face_size: Optional[int] = None
    min_cx_ratio: Optional[float] = None
    max_cx_ratio: Optional[float] = None
    min_cy_ratio: Optional[float] = None
    max_cy_ratio: Optional[float] = None
    min_yaw_ratio: Optional[float] = None
    max_yaw_ratio: Optional[float] = None
    min_pitch_ratio: Optional[float] = None
    max_pitch_ratio: Optional[float] = None
    min_quality_score: Optional[float] = None
    similarity_threshold: Optional[float] = None

@app.get("/config")
def get_config(pipeline: FacePipeline = Depends(get_pipeline)):
    return {
        "min_face_size": pipeline.min_face_size,
        "min_cx_ratio": pipeline.min_cx_ratio,
        "max_cx_ratio": pipeline.max_cx_ratio,
        "min_cy_ratio": pipeline.min_cy_ratio,
        "max_cy_ratio": pipeline.max_cy_ratio,
        "min_yaw_ratio": pipeline.min_yaw_ratio,
        "max_yaw_ratio": pipeline.max_yaw_ratio,
        "min_pitch_ratio": pipeline.min_pitch_ratio,
        "max_pitch_ratio": pipeline.max_pitch_ratio,
        "min_quality_score": pipeline.min_quality_score,
        "similarity_threshold": pipeline.similarity_threshold,
    }

@app.post("/config")
def update_config(cfg: ConfigUpdate, pipeline: FacePipeline = Depends(get_pipeline)):
    if cfg.min_face_size is not None: pipeline.min_face_size = cfg.min_face_size
    if cfg.min_cx_ratio is not None: pipeline.min_cx_ratio = cfg.min_cx_ratio
    if cfg.max_cx_ratio is not None: pipeline.max_cx_ratio = cfg.max_cx_ratio
    if cfg.min_cy_ratio is not None: pipeline.min_cy_ratio = cfg.min_cy_ratio
    if cfg.max_cy_ratio is not None: pipeline.max_cy_ratio = cfg.max_cy_ratio
    if cfg.min_yaw_ratio is not None: pipeline.min_yaw_ratio = cfg.min_yaw_ratio
    if cfg.max_yaw_ratio is not None: pipeline.max_yaw_ratio = cfg.max_yaw_ratio
    if cfg.min_pitch_ratio is not None: pipeline.min_pitch_ratio = cfg.min_pitch_ratio
    if cfg.max_pitch_ratio is not None: pipeline.max_pitch_ratio = cfg.max_pitch_ratio
    if cfg.min_quality_score is not None: pipeline.min_quality_score = cfg.min_quality_score
    if cfg.similarity_threshold is not None: pipeline.similarity_threshold = cfg.similarity_threshold
    return {"status": "success", "config": get_config(pipeline)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/identify")
async def identify(
    photo: UploadFile = File(..., description="Face photo to identify"),
    pipeline: FacePipeline = Depends(get_pipeline),
    db: EmployeeDB = Depends(get_db),
):
    """
    Identify a person from a photo by comparing their embedding with the database.
    """
    img_bytes = await photo.read()
    try:
        results = pipeline.process_image_bytes(img_bytes, max_faces=1)
        query_embed = results[0]["embed"]
        
        # Fetch all embeddings from DB
        db_embeddings = db.get_all_embeddings()
        if not db_embeddings:
            raise HTTPException(400, "Database is empty — cannot perform identification")
            
        import numpy as np
        best_score = -1.0
        best_match = None
        
        for record in db_embeddings:
            score = float(np.dot(query_embed, record["embed"]))
            if score > best_score:
                best_score = score
                best_match = record
                
        if best_score >= pipeline.similarity_threshold:
            return {
                "recognized": True,
                "person_id": best_match["person_id"],
                "name": best_match["name"],
                "similarity_score": best_score,
                "threshold": pipeline.similarity_threshold
            }
        else:
            return {
                "recognized": False,
                "similarity_score": best_score,
                "threshold": pipeline.similarity_threshold,
                "message": f"Best match score ({best_score:.4f}) is below threshold ({pipeline.similarity_threshold})"
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(422, f"Failed to identify face: {e}")


@app.get("/stats", response_model=StatsOut)
def stats(db: EmployeeDB = Depends(get_db)):
    return db.stats()


# ── Persons ──────────────────────────────────────────────────────────────────

@app.get("/persons", response_model=List[PersonOut])
def list_persons(db: EmployeeDB = Depends(get_db)):
    return db.list_persons()


@app.get("/persons/{person_id}", response_model=PersonOut)
def get_person(person_id: int, db: EmployeeDB = Depends(get_db)):
    p = db.get_person(person_id)
    if not p:
        raise HTTPException(404, f"Person {person_id} not found")
    return p


@app.delete("/persons/{person_id}")
def delete_person(person_id: int, db: EmployeeDB = Depends(get_db)):
    if not db.delete_person(person_id):
        raise HTTPException(404, f"Person {person_id} not found")
    return {"deleted": person_id}


# ── Enrollment ────────────────────────────────────────────────────────────────

@app.post("/enroll", response_model=EnrollResult)
async def enroll(
    name: str = Form(..., description="Employee full name"),
    department: str = Form("", description="Department (optional)"),
    photos: List[UploadFile] = File(..., description="1 or more face photos"),
    pipeline: FacePipeline = Depends(get_pipeline),
    db: EmployeeDB = Depends(get_db),
):
    """
    Enroll a new employee.
    - Detects face in each photo
    - Aligns (Umeyama 112×112)
    - Computes ArcFace embedding (L2-normalized 512-dim)
    - Stores in employee.db
    """
    person_id = db.add_person(name)
    processed = 0
    failed = []

    for photo in photos:
        img_bytes = await photo.read()
        try:
            results = pipeline.process_image_bytes(img_bytes, max_faces=1)
            embed = results[0]["embed"]
            db.add_embedding(person_id, embed)
            processed += 1
        except Exception as e:
            failed.append(f"{photo.filename}: {e}")

    if processed == 0:
        db.delete_person(person_id)
        raise HTTPException(422, f"No faces detected in any photo. Errors: {failed}")

    embeds = db.get_embeddings(person_id)
    return EnrollResult(
        person_id=person_id,
        name=name,
        photos_processed=processed,
        photos_failed=len(failed),
        embed_count=len(embeds),
        failed_files=failed,
    )


@app.post("/persons/{person_id}/photos", response_model=EnrollResult)
async def add_photos(
    person_id: int,
    photos: List[UploadFile] = File(...),
    pipeline: FacePipeline = Depends(get_pipeline),
    db: EmployeeDB = Depends(get_db),
):
    """Add more photos to an existing person."""
    p = db.get_person(person_id)
    if not p:
        raise HTTPException(404, f"Person {person_id} not found")

    processed = 0
    failed = []
    for photo in photos:
        img_bytes = await photo.read()
        try:
            results = pipeline.process_image_bytes(img_bytes, max_faces=1)
            db.add_embedding(person_id, results[0]["embed"])
            processed += 1
        except Exception as e:
            failed.append(f"{photo.filename}: {e}")

    embeds = db.get_embeddings(person_id)
    return EnrollResult(
        person_id=person_id,
        name=p["name"],
        photos_processed=processed,
        photos_failed=len(failed),
        embed_count=len(embeds),
        failed_files=failed,
    )


@app.delete("/persons/{person_id}/embeddings/{embed_id}")
def delete_embedding(person_id: int, embed_id: int, db: EmployeeDB = Depends(get_db)):
    if not db.delete_embedding(embed_id):
        raise HTTPException(404, "Embedding not found")
    return {"deleted": embed_id}


# ── Database export ───────────────────────────────────────────────────────────

@app.get("/database/export")
def export_db(db: EmployeeDB = Depends(get_db)):
    """
    Download employee.db — ready to copy to camera at /sdcard/employee.db
    Camera command: cp /sdcard/nfs/employee.db /sdcard/employee.db && killall cv_alg
    """
    stats = db.stats()
    if stats["persons"] == 0:
        raise HTTPException(400, "Database is empty — enroll at least one person first")

    return FileResponse(
        path=DB_PATH,
        filename="employee.db",
        media_type="application/octet-stream",
        headers={
            "X-Persons": str(stats["persons"]),
            "X-Embeddings": str(stats["embeddings"]),
            "X-Size-KB": str(stats["size_kb"]),
        }
    )


@app.post("/database/reset")
def reset_db(db: EmployeeDB = Depends(get_db)):
    """Delete all persons and embeddings (irreversible)."""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("DELETE FROM embedding; DELETE FROM person; DELETE FROM sqlite_sequence;")
    conn.commit()
    conn.close()
    # Re-init state
    if hasattr(app.state, "db"):
        delattr(app.state, "db")
    return {"reset": True}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
