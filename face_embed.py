#!/usr/bin/env python3
"""
face_embed.py — Extract a 512-dim ArcFace embedding from a single photo.

Models are auto-detected from the models/ directory next to this script.

Usage:
    from face_embed import extract_embedding

    # Simcam — lightweight model (default)
    result = extract_embedding("photo.jpg")
    result = extract_embedding("photo.jpg", recognizer="w600k_mbf")

    # AIBOX — stronger model
    result = extract_embedding("photo.jpg", recognizer="w600k_r50")
    result = extract_embedding("photo.jpg", recognizer="glintr100")

    # Raw bytes (HTTP upload)
    result = extract_embedding(request.body, recognizer="w600k_r50")

    if result["ok"]:
        embedding  = result["embedding"]    # list[float], 512 values
        quality    = result["quality"]      # float, 0.0 – 1.0
        recognizer = result["recognizer"]   # str, model name actually used
    else:
        error = result["error"]             # str, reason

Available recognizer values:
    "w600k_mbf"  — MobileFaceNet,  13 MB,  fast    (Simcam / CV25 NPU)
    "w600k_r50"  — ResNet-50,     167 MB,  strong  (AIBOX)
    "glintr100"  — ResNet-100,    249 MB,  best    (AIBOX, Glint360K dataset)
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

# ── Global model cache ────────────────────────────────────────────────────────
# SCRFD and CR-FIQA: single shared instance
_scrfd        = None
_crfiqa       = None
_scrfd_input  = None
_crfiqa_input = None

# ArcFace: one entry per loaded model  { model_path: (session, input_name) }
_arcface_cache: dict = {}

# Recognizer name → resolved model path  { "w600k_mbf": "/path/w600k_mbf.onnx" }
_recognizer_index: dict = {}


# ── Model discovery ───────────────────────────────────────────────────────────

# Keywords that identify each model role by filename
_SCRFD_KEYS    = ("scrfd",)
_ARCFACE_KEYS  = ("mbf", "w600k", "arcface", "r50", "r100",
                  "buffalo", "antelope", "glint", "glintr")
_CRFIQA_KEYS   = ("crfiqa", "quality", "fiqa")


def _scan_models(models_dir: str) -> dict:
    """
    Scan models/ directory and return a categorized index.

    Returns:
        {
            "scrfd":  "/path/scrfd_10g.onnx",      # single path or None
            "crfiqa": "/path/crfiqa.onnx",          # single path or None
            "arcface": {                            # all recognition models found
                "w600k_mbf": "/path/w600k_mbf.onnx",
                "w600k_r50": "/path/w600k_r50.onnx",
                "glintr100": "/path/glintr100.onnx",
            }
        }
    """
    index = {"scrfd": None, "crfiqa": None, "arcface": {}}

    for f in sorted(glob.glob(os.path.join(models_dir, "*.onnx"))):
        stem = os.path.splitext(os.path.basename(f))[0].lower()

        if any(k in stem for k in _SCRFD_KEYS) and index["scrfd"] is None:
            index["scrfd"] = f
        elif any(k in stem for k in _CRFIQA_KEYS) and index["crfiqa"] is None:
            index["crfiqa"] = f
        elif any(k in stem for k in _ARCFACE_KEYS):
            # Key = filename without extension, lowercased
            index["arcface"][stem] = f

    return index


def _resolve_recognizer(name: str | None, arcface_map: dict) -> tuple[str, str]:
    """
    Resolve a recognizer name to (key, model_path).

    Args:
        name       : requested recognizer name, or None for auto-select
        arcface_map: {"stem": "path"} dict from _scan_models()

    Returns:
        (resolved_key, model_path)

    Raises:
        RuntimeError if no match found
    """
    if not arcface_map:
        raise RuntimeError(
            f"No ArcFace recognition model found in: {_MODELS_DIR}/\n"
            f"  Expected files like: w600k_mbf.onnx, w600k_r50.onnx, glintr100.onnx"
        )

    if name is None:
        # Auto-select: prefer mbf (Simcam default), else first found
        for preferred in ("w600k_mbf", "mbf"):
            if preferred in arcface_map:
                return preferred, arcface_map[preferred]
        key = next(iter(arcface_map))
        return key, arcface_map[key]

    # Exact match first
    stem = os.path.splitext(name.lower())[0]   # strip .onnx if present
    if stem in arcface_map:
        return stem, arcface_map[stem]

    # Partial match (e.g. "r50" matches "w600k_r50")
    matches = [(k, v) for k, v in arcface_map.items() if stem in k]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous recognizer '{name}' matches: {[k for k, _ in matches]}\n"
            f"  Use a more specific name."
        )

    raise RuntimeError(
        f"Recognizer '{name}' not found in: {_MODELS_DIR}/\n"
        f"  Available: {list(arcface_map.keys())}"
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def _ensure_shared_models():
    """Load SCRFD and CR-FIQA once (shared across all recognizer calls)."""
    global _scrfd, _crfiqa, _scrfd_input, _crfiqa_input

    if _scrfd is not None:
        return

    index = _scan_models(_MODELS_DIR)

    if index["scrfd"] is None:
        raise RuntimeError(
            f"SCRFD detector not found in: {_MODELS_DIR}/\n"
            f"  Expected file like: scrfd_10g_bnkps.onnx"
        )

    providers = ["CPUExecutionProvider"]
    _scrfd       = ort.InferenceSession(index["scrfd"],  providers=providers)
    _scrfd_input = _scrfd.get_inputs()[0].name

    if index["crfiqa"]:
        _crfiqa       = ort.InferenceSession(index["crfiqa"], providers=providers)
        _crfiqa_input = _crfiqa.get_inputs()[0].name

    # Build recognizer name index once
    _recognizer_index.update(index["arcface"])


def _ensure_arcface(recognizer: str | None) -> tuple:
    """
    Return (session, input_name) for the requested recognizer.
    Loads and caches the model on first use; subsequent calls are instant.
    """
    _ensure_shared_models()

    key, model_path = _resolve_recognizer(recognizer, _recognizer_index)

    if model_path not in _arcface_cache:
        _arcface_cache[model_path] = (
            ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]),
            None,  # placeholder
        )
        session = _arcface_cache[model_path][0]
        _arcface_cache[model_path] = (session, session.get_inputs()[0].name)

    return key, *_arcface_cache[model_path]


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

def _embed(face_bgr: np.ndarray, session, input_name: str) -> np.ndarray:
    """ArcFace: 112×112 BGR → L2-normalized 512-dim float32 embedding vector."""
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 127.5, (_ARCFACE_SIZE, _ARCFACE_SIZE),
        (127.5, 127.5, 127.5), swapRB=True,
    )
    raw  = session.run(None, {input_name: blob})[0][0]
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
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_embedding(image_source, recognizer: str = None) -> dict:
    """
    Extract a 512-dim ArcFace embedding from a single face photo.

    Args:
        image_source : str   — file path ("photo.jpg")
                     | bytes — raw bytes from an HTTP upload

        recognizer   : str | None — recognition model to use.
                       None        → auto-select (prefers w600k_mbf, Simcam default)
                       "w600k_mbf" → MobileFaceNet, 13 MB  (Simcam / CV25 NPU)
                       "w600k_r50" → ResNet-50,    167 MB  (AIBOX)
                       "glintr100" → ResNet-100,   249 MB  (AIBOX, highest accuracy)

                       Partial names also work: "r50" → "w600k_r50", "r100" → "glintr100"

    Returns:
        On success:
            {
                "ok"        : True,
                "embedding" : [float, ...],  # 512 values, L2-normalized
                "quality"   : float,         # CR-FIQA score, 0.0 – 1.0
                "recognizer": str            # model name actually used
            }
        On failure:
            {
                "ok"   : False,
                "error": str    # human-readable reason
            }
    """
    # Load shared models (SCRFD + CR-FIQA) and resolve recognizer
    try:
        rec_key, arc_session, arc_input = _ensure_arcface(recognizer)
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
        emb = _embed(aligned, arc_session, arc_input)
    except Exception as e:
        return {"ok": False, "error": f"Embedding error: {e}"}

    return {
        "ok":         True,
        "embedding":  emb.tolist(),   # list[float], 512 values
        "quality":    round(q, 6),
        "recognizer": rec_key,        # model name actually used
    }


def list_recognizers() -> list[str]:
    """
    Return names of all ArcFace recognition models available in models/.

    Example:
        ["w600k_mbf", "w600k_r50", "glintr100"]
    """
    index = _scan_models(_MODELS_DIR)
    return list(index["arcface"].keys())
