#!/usr/bin/env python3
"""
face_embed.py — Extract a 512-dim ArcFace embedding from a single photo.

Models are auto-detected from the models/ directory next to this script.

Usage:
    from face_embed import extract_embedding

    result = extract_embedding("photo.jpg")   # file path
    result = extract_embedding(image_bytes)   # raw bytes from HTTP upload

    if result["ok"]:
        embedding = result["embedding"]   # list[float], 512 values
        quality   = result["quality"]     # float, 0.0 – 1.0
    else:
        error = result["error"]           # str, reason
"""

import glob
import os

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

# ── Model paths (auto-detected from models/ directory next to this script) ────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── ArcFace standard 5-point reference template (112×112) ────────────────────
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ── Guard thresholds ──────────────────────────────────────────────────────────
_MIN_FACE_SIZE    = 80      # px — minimum face width and height
_MIN_CX_RATIO     = 0.25    # face center x ratio relative to image width
_MAX_CX_RATIO     = 0.75
_MIN_CY_RATIO     = 0.25    # face center y ratio relative to image height
_MAX_CY_RATIO     = 0.75
_MIN_YAW_RATIO    = 0.5     # left/right turn symmetry ratio
_MAX_YAW_RATIO    = 2.0
_MIN_PITCH_RATIO  = 0.4     # up/down tilt symmetry ratio
_MAX_PITCH_RATIO  = 1.8
_MIN_QUALITY      = 0.15    # CR-FIQA minimum quality score

# SCRFD / ArcFace inference parameters
_SCRFD_SIZE         = 640
_ARCFACE_SIZE       = 112
_DETECT_THRESHOLD   = 0.30
_NMS_THRESHOLD      = 0.40

# ── Global model cache (loaded once, reused on subsequent calls) ──────────────
_scrfd   = None
_arcface = None
_crfiqa  = None
_scrfd_input   = None
_arcface_input = None
_crfiqa_input  = None


def _find_models(models_dir: str) -> dict:
    """Locate ONNX models in models/ directory by filename pattern."""
    found = {}
    for f in sorted(glob.glob(os.path.join(models_dir, "*.onnx"))):
        n = os.path.basename(f).lower()
        if "scrfd" in n and "scrfd" not in found:
            found["scrfd"] = f
        elif any(k in n for k in ("mbf", "w600k", "arcface", "r50", "r100")) \
                and "arcface" not in found:
            found["arcface"] = f
        elif any(k in n for k in ("crfiqa", "quality", "fiqa")) \
                and "crfiqa" not in found:
            found["crfiqa"] = f
    return found


def _load_models():
    """Load models once into global variables; subsequent calls are no-ops."""
    global _scrfd, _arcface, _crfiqa
    global _scrfd_input, _arcface_input, _crfiqa_input

    if _scrfd is not None:
        return  # already loaded

    models = _find_models(_MODELS_DIR)
    providers = ["CPUExecutionProvider"]

    for key in ("scrfd", "arcface"):
        if key not in models:
            raise RuntimeError(
                f"'{key}' model not found in: {_MODELS_DIR}/\n"
                f"  Expected files: scrfd*.onnx, w600k*.onnx"
            )

    _scrfd   = ort.InferenceSession(models["scrfd"],   providers=providers)
    _arcface = ort.InferenceSession(models["arcface"], providers=providers)
    _scrfd_input   = _scrfd.get_inputs()[0].name
    _arcface_input = _arcface.get_inputs()[0].name

    if "crfiqa" in models:
        _crfiqa = ort.InferenceSession(models["crfiqa"], providers=providers)
        _crfiqa_input = _crfiqa.get_inputs()[0].name


# ── Face detection (SCRFD) ────────────────────────────────────────────────────

def _letterbox(img: np.ndarray) -> tuple:
    """Resize image to 640×640 with top-left letterbox padding for SCRFD input."""
    h, w = img.shape[:2]
    ratio = float(h) / w
    if ratio > 1.0:
        new_h, new_w = _SCRFD_SIZE, int(_SCRFD_SIZE / ratio)
    else:
        new_w, new_h = _SCRFD_SIZE, int(_SCRFD_SIZE * ratio)
    scale = float(new_h) / h
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((_SCRFD_SIZE, _SCRFD_SIZE, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def _detect(img_bgr: np.ndarray) -> list:
    """
    Run SCRFD face detector. Applies NMS and sorts results by score descending.
    Returns: [{"bbox": [x1,y1,x2,y2], "kps5": ndarray(5,2), "score": float}]
    """
    h, w = img_bgr.shape[:2]
    lb, scale = _letterbox(img_bgr)

    blob = cv2.dnn.blobFromImage(
        lb, 1.0 / 128.0, (_SCRFD_SIZE, _SCRFD_SIZE),
        (127.5, 127.5, 127.5), swapRB=True,
    )
    outs = _scrfd.run(None, {_scrfd_input: blob})

    strides    = [8, 16, 32]
    feat_sizes = [_SCRFD_SIZE // s for s in strides]
    results    = []

    for si, stride in enumerate(strides):
        fs     = feat_sizes[si]
        scores = outs[si].reshape(-1)
        bboxes = outs[3 + si].reshape(-1, 4)
        kpss   = outs[6 + si].reshape(-1, 5, 2)

        cx = np.tile(np.repeat(np.arange(fs), 2), fs) * stride
        cy = np.repeat(np.arange(fs) * stride, fs * 2)

        for i in range(len(scores)):
            sc = float(scores[i])
            if sc < _DETECT_THRESHOLD:
                continue
            dx, dy, dw, dh = bboxes[i]
            x1 = (cx[i] - dx * stride) / scale
            y1 = (cy[i] - dy * stride) / scale
            x2 = (cx[i] + dw * stride) / scale
            y2 = (cy[i] + dh * stride) / scale
            kps = np.array([
                [(cx[i] + kp[0] * stride) / scale,
                 (cy[i] + kp[1] * stride) / scale]
                for kp in kpss[i]
            ], dtype=np.float32)
            results.append({"bbox": [x1, y1, x2, y2], "kps5": kps, "score": sc})

    if len(results) > 1:
        boxes  = [[r["bbox"][0], r["bbox"][1],
                   r["bbox"][2] - r["bbox"][0],
                   r["bbox"][3] - r["bbox"][1]] for r in results]
        scores = [r["score"] for r in results]
        idx    = cv2.dnn.NMSBoxes(boxes, scores, _DETECT_THRESHOLD, _NMS_THRESHOLD)
        results = [results[i] for i in idx.flatten()]

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Face alignment (Umeyama) ──────────────────────────────────────────────────

def _align(img_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
    """Apply Umeyama similarity transform to produce a 112×112 aligned face chip."""
    tform = SimilarityTransform()
    tform.estimate(kps5, _ARCFACE_DST)
    return cv2.warpAffine(img_bgr, tform.params[:2],
                          (_ARCFACE_SIZE, _ARCFACE_SIZE),
                          flags=cv2.INTER_LINEAR)


# ── Embedding (ArcFace) ───────────────────────────────────────────────────────

def _embed(face_bgr: np.ndarray) -> np.ndarray:
    """ArcFace: 112×112 BGR → L2-normalized 512-dim float32 embedding vector."""
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 127.5, (_ARCFACE_SIZE, _ARCFACE_SIZE),
        (127.5, 127.5, 127.5), swapRB=True,
    )
    raw  = _arcface.run(None, {_arcface_input: blob})[0][0]
    norm = np.linalg.norm(raw)
    return (raw / norm).astype(np.float32) if norm > 1e-9 else raw.astype(np.float32)


# ── Biometric quality score (CR-FIQA) ────────────────────────────────────────

def _quality(face_bgr: np.ndarray) -> float:
    """Return CR-FIQA quality score (0.0–1.0). Returns 1.0 if model is not loaded."""
    if _crfiqa is None:
        return 1.0
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 128.0, (_ARCFACE_SIZE, _ARCFACE_SIZE),
        (127.5, 127.5, 127.5), swapRB=True,
    )
    return float(_crfiqa.run(None, {_crfiqa_input: blob})[0][0][0])


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — the only function the web backend needs
# ══════════════════════════════════════════════════════════════════════════════

def extract_embedding(image_source) -> dict:
    """
    Extract a 512-dim ArcFace embedding from a single face photo.

    Args:
        image_source : str   — file path ("photo.jpg")
                     | bytes — raw bytes from an HTTP upload

    Returns:
        On success:
            {
                "ok"       : True,
                "embedding": [float, ...],  # 512 values, L2-normalized
                "quality"  : float          # CR-FIQA score, 0.0 – 1.0
            }
        On failure:
            {
                "ok"   : False,
                "error": str    # human-readable reason
            }

    Possible failure reasons (ok=False):
        - Image could not be read
        - No face detected
        - Multiple faces detected
        - Face too small (< 80px)
        - Face not centered in the frame
        - Head rotated too far (yaw / pitch out of range)
        - CR-FIQA quality score too low (< 0.15)
    """
    # Load models on first call (no-op on subsequent calls)
    try:
        _load_models()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}

    # Decode image
    try:
        if isinstance(image_source, (bytes, bytearray)):
            arr = np.frombuffer(image_source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(image_source))

        if img is None:
            return {"ok": False, "error": "Cannot read image (invalid path or corrupted file)"}
    except Exception as e:
        return {"ok": False, "error": f"Image load error: {e}"}

    H, W = img.shape[:2]

    # ── Face detection ────────────────────────────────────────────────────────
    try:
        faces = _detect(img)
    except Exception as e:
        return {"ok": False, "error": f"Detection error: {e}"}

    if not faces:
        return {"ok": False, "error": "No face detected in image"}

    if len(faces) > 1:
        return {"ok": False,
                "error": f"Multiple faces detected ({len(faces)}), only 1 face allowed per photo"}

    face = faces[0]
    bbox = face["bbox"]
    kps5 = face["kps5"]

    # ── Guard 1: Face size ────────────────────────────────────────────────────
    fw = bbox[2] - bbox[0]
    fh = bbox[3] - bbox[1]
    if fw < _MIN_FACE_SIZE or fh < _MIN_FACE_SIZE:
        return {"ok": False,
                "error": f"Face too small ({int(fw)}x{int(fh)}px), "
                         f"minimum is {_MIN_FACE_SIZE}x{_MIN_FACE_SIZE}px"}

    # ── Guard 2: Face centrality ──────────────────────────────────────────────
    cx_r = ((bbox[0] + bbox[2]) / 2.0) / W
    cy_r = ((bbox[1] + bbox[3]) / 2.0) / H
    if not (_MIN_CX_RATIO <= cx_r <= _MAX_CX_RATIO
            and _MIN_CY_RATIO <= cy_r <= _MAX_CY_RATIO):
        return {"ok": False,
                "error": f"Face not centered (cx={cx_r:.2f}, cy={cy_r:.2f}), "
                         f"expected center within 0.25–0.75 of image"}

    # ── Guard 3: Yaw (left/right rotation) ───────────────────────────────────
    d_left  = np.linalg.norm(kps5[2] - kps5[0])
    d_right = np.linalg.norm(kps5[2] - kps5[1])
    yaw     = max(d_left, 1e-5) / max(d_right, 1e-5)
    if not (_MIN_YAW_RATIO <= yaw <= _MAX_YAW_RATIO):
        return {"ok": False,
                "error": f"Face turned too far sideways (yaw ratio={yaw:.2f}), "
                         f"please look directly at camera"}

    # ── Guard 4: Pitch (up/down tilt) ────────────────────────────────────────
    eye_mid   = (kps5[0] + kps5[1]) / 2.0
    mouth_mid = (kps5[3] + kps5[4]) / 2.0
    pitch     = (max(np.linalg.norm(kps5[2] - eye_mid),   1e-5)
               / max(np.linalg.norm(kps5[2] - mouth_mid), 1e-5))
    if not (_MIN_PITCH_RATIO <= pitch <= _MAX_PITCH_RATIO):
        return {"ok": False,
                "error": f"Face tilted too far up/down (pitch ratio={pitch:.2f}), "
                         f"please keep face level"}

    # ── Alignment → 112×112 ──────────────────────────────────────────────────
    try:
        aligned = _align(img, kps5)
    except Exception as e:
        return {"ok": False, "error": f"Alignment error: {e}"}

    # ── Guard 5: CR-FIQA quality score ────────────────────────────────────────
    try:
        q = _quality(aligned)
    except Exception as e:
        return {"ok": False, "error": f"Quality check error: {e}"}

    if q < _MIN_QUALITY:
        return {"ok": False,
                "error": f"Face quality too low (score={q:.4f}), "
                         f"minimum is {_MIN_QUALITY:.2f}. Use a clearer photo"}

    # ── ArcFace embedding ─────────────────────────────────────────────────────
    try:
        emb = _embed(aligned)
    except Exception as e:
        return {"ok": False, "error": f"Embedding error: {e}"}

    return {
        "ok":        True,
        "embedding": emb.tolist(),   # list[float], 512 values
        "quality":   round(q, 6),
    }


