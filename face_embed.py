#!/usr/bin/env python3
"""
face_embed.py — Rasmdan 512-o'lchamli ArcFace embedding ajratadi.

Modellar: skript yonidagi models/ papkasidan avtomatik topiladi.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNKSIYA sifatida (web backend import qiladi):

    from face_embed import extract_embedding

    result = extract_embedding("photo.jpg")

    if result["ok"]:
        embedding = result["embedding"]   # list[float], 512 ta
        quality   = result["quality"]     # float, 0.0 – 1.0
    else:
        error = result["error"]           # str, sabab

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI sifatida (subprocess orqali):

    python3 face_embed.py photo.jpg

    # Muvaffaqiyat (stdout):
    # {"ok": true, "embedding": [0.021, -0.043, ...], "quality": 0.724}

    # Xato (stdout):
    # {"ok": false, "error": "No face detected in image"}

    Exit code: 0 = muvaffaqiyat, 1 = xato
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

# ── Model yo'llari (skript yonidagi models/ dan avtomatik topiladi) ───────────
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── ArcFace 5-nuqtali standart shablon (112×112) ─────────────────────────────
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ── Guard chegaralari ─────────────────────────────────────────────────────────
_MIN_FACE_SIZE    = 80      # px — yuzning minimal eni/bo'yi
_MIN_CX_RATIO     = 0.25    # yuz markazi x koordinatasi (rasmga nisbatan)
_MAX_CX_RATIO     = 0.75
_MIN_CY_RATIO     = 0.25    # yuz markazi y koordinatasi
_MAX_CY_RATIO     = 0.75
_MIN_YAW_RATIO    = 0.5     # chapga/o'ngga burilish simmetriyasi
_MAX_YAW_RATIO    = 2.0
_MIN_PITCH_RATIO  = 0.4     # tepaga/pastga egilish simmetriyasi
_MAX_PITCH_RATIO  = 1.8
_MIN_QUALITY      = 0.15    # CR-FIQA minimal sifat bahosi

# SCRFD / ArcFace inference parametrlari
_SCRFD_SIZE         = 640
_ARCFACE_SIZE       = 112
_DETECT_THRESHOLD   = 0.30
_NMS_THRESHOLD      = 0.40

# ── Global model cache (bir marta yuklanadi) ──────────────────────────────────
_scrfd   = None
_arcface = None
_crfiqa  = None
_scrfd_input   = None
_arcface_input = None
_crfiqa_input  = None


def _find_models(models_dir: str) -> dict:
    """models/ papkasidagi ONNX fayllarni fayl nomi bo'yicha topadi."""
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
    """Modellarni bir marta yuklab global o'zgaruvchilarga saqlaydi."""
    global _scrfd, _arcface, _crfiqa
    global _scrfd_input, _arcface_input, _crfiqa_input

    if _scrfd is not None:
        return  # allaqachon yuklangan

    models = _find_models(_MODELS_DIR)
    providers = ["CPUExecutionProvider"]

    for key in ("scrfd", "arcface"):
        if key not in models:
            raise RuntimeError(
                f"'{key}' modeli topilmadi: {_MODELS_DIR}/\n"
                f"  Kutilayotgan fayllar: scrfd*.onnx, w600k*.onnx"
            )

    _scrfd   = ort.InferenceSession(models["scrfd"],   providers=providers)
    _arcface = ort.InferenceSession(models["arcface"], providers=providers)
    _scrfd_input   = _scrfd.get_inputs()[0].name
    _arcface_input = _arcface.get_inputs()[0].name

    if "crfiqa" in models:
        _crfiqa = ort.InferenceSession(models["crfiqa"], providers=providers)
        _crfiqa_input = _crfiqa.get_inputs()[0].name


# ── Yuz aniqlash (SCRFD) ──────────────────────────────────────────────────────

def _letterbox(img: np.ndarray) -> tuple:
    """Rasmni SCRFD uchun 640×640 ga letterbox qiladi (top-left padding)."""
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
    SCRFD yuz aniqlovchi. NMS qo'llaydi, score bo'yicha tartiblaydi.
    Qaytaradi: [{"bbox": [x1,y1,x2,y2], "kps5": ndarray(5,2), "score": float}]
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


# ── Hizalash (Umeyama) ────────────────────────────────────────────────────────

def _align(img_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
    """Umeyama o'xshashlik transformatsiyasi → 112×112 BGR yuz chipi."""
    tform = SimilarityTransform()
    tform.estimate(kps5, _ARCFACE_DST)
    return cv2.warpAffine(img_bgr, tform.params[:2],
                          (_ARCFACE_SIZE, _ARCFACE_SIZE),
                          flags=cv2.INTER_LINEAR)


# ── Embedding (ArcFace) ───────────────────────────────────────────────────────

def _embed(face_bgr: np.ndarray) -> np.ndarray:
    """ArcFace: 112×112 BGR → L2-normallashtirilgan 512-dim float32 vektor."""
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 127.5, (_ARCFACE_SIZE, _ARCFACE_SIZE),
        (127.5, 127.5, 127.5), swapRB=True,
    )
    raw  = _arcface.run(None, {_arcface_input: blob})[0][0]
    norm = np.linalg.norm(raw)
    return (raw / norm).astype(np.float32) if norm > 1e-9 else raw.astype(np.float32)


# ── Sifat bahosi (CR-FIQA) ────────────────────────────────────────────────────

def _quality(face_bgr: np.ndarray) -> float:
    """CR-FIQA sifat bahosi (0.0 – 1.0). Model yo'q bo'lsa 1.0 qaytaradi."""
    if _crfiqa is None:
        return 1.0
    blob = cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 128.0, (_ARCFACE_SIZE, _ARCFACE_SIZE),
        (127.5, 127.5, 127.5), swapRB=True,
    )
    return float(_crfiqa.run(None, {_crfiqa_input: blob})[0][0][0])


# ══════════════════════════════════════════════════════════════════════════════
# ASOSIY FUNKSIYA — web team ishlatadigan yagona interfeys
# ══════════════════════════════════════════════════════════════════════════════

def extract_embedding(image_source) -> dict:
    """
    Rasmdan 512-o'lchamli ArcFace embedding ajratadi.

    Parametr:
        image_source : str  — fayl yo'li  ("photo.jpg")
                     | bytes — xom baytlar (HTTP upload dan)

    Qaytaradi (dict):
        Muvaffaqiyat:
            {
                "ok"       : True,
                "embedding": [float, ...],  # 512 ta, L2-normallashtirilgan
                "quality"  : float          # CR-FIQA bahosi, 0.0 – 1.0
            }
        Xato:
            {
                "ok"   : False,
                "error": str    # sabab (inglizcha, logging uchun qulay)
            }

    Xatolar (ok=False bo'lishi mumkin bo'lgan holatlar):
        - Rasm o'qilmadi
        - Yuz topilmadi
        - Bir nechta yuz topildi
        - Yuz juda kichik (< 80px)
        - Yuz markazda emas
        - Bosh juda ko'p burilgan (yaw / pitch)
        - CR-FIQA sifat bahosi juda past (< 0.15)
    """
    # Modellarni yuklash (birinchi chaqiruvda)
    try:
        _load_models()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}

    # Rasmni o'qish
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

    # ── Yuz aniqlash ─────────────────────────────────────────────────────────
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

    # ── Guard 1: Yuz o'lchami ─────────────────────────────────────────────────
    fw = bbox[2] - bbox[0]
    fh = bbox[3] - bbox[1]
    if fw < _MIN_FACE_SIZE or fh < _MIN_FACE_SIZE:
        return {"ok": False,
                "error": f"Face too small ({int(fw)}x{int(fh)}px), "
                         f"minimum is {_MIN_FACE_SIZE}x{_MIN_FACE_SIZE}px"}

    # ── Guard 2: Markaziylik ──────────────────────────────────────────────────
    cx_r = ((bbox[0] + bbox[2]) / 2.0) / W
    cy_r = ((bbox[1] + bbox[3]) / 2.0) / H
    if not (_MIN_CX_RATIO <= cx_r <= _MAX_CX_RATIO
            and _MIN_CY_RATIO <= cy_r <= _MAX_CY_RATIO):
        return {"ok": False,
                "error": f"Face not centered (cx={cx_r:.2f}, cy={cy_r:.2f}), "
                         f"expected center within 0.25–0.75 of image"}

    # ── Guard 3: Yaw (chapga/o'ngga burilish) ────────────────────────────────
    d_left  = np.linalg.norm(kps5[2] - kps5[0])
    d_right = np.linalg.norm(kps5[2] - kps5[1])
    yaw     = max(d_left, 1e-5) / max(d_right, 1e-5)
    if not (_MIN_YAW_RATIO <= yaw <= _MAX_YAW_RATIO):
        return {"ok": False,
                "error": f"Face turned too far sideways (yaw ratio={yaw:.2f}), "
                         f"please look directly at camera"}

    # ── Guard 4: Pitch (tepaga/pastga egilish) ────────────────────────────────
    eye_mid   = (kps5[0] + kps5[1]) / 2.0
    mouth_mid = (kps5[3] + kps5[4]) / 2.0
    pitch     = (max(np.linalg.norm(kps5[2] - eye_mid),   1e-5)
               / max(np.linalg.norm(kps5[2] - mouth_mid), 1e-5))
    if not (_MIN_PITCH_RATIO <= pitch <= _MAX_PITCH_RATIO):
        return {"ok": False,
                "error": f"Face tilted too far up/down (pitch ratio={pitch:.2f}), "
                         f"please keep face level"}

    # ── Hizalash → 112×112 ────────────────────────────────────────────────────
    try:
        aligned = _align(img, kps5)
    except Exception as e:
        return {"ok": False, "error": f"Alignment error: {e}"}

    # ── Guard 5: CR-FIQA sifat bahosi ─────────────────────────────────────────
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
        "embedding": emb.tolist(),   # list[float], 512 ta
        "quality":   round(q, 6),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python3 face_embed.py <image_path>', file=sys.stderr)
        sys.exit(1)

    result = extract_embedding(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["ok"] else 1)
