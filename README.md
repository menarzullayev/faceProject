# face_embed.py

Extract a **512-dim ArcFace embedding** from a single face photo.  
Single self-contained Python file — no server, no API, no database.

---

## How It Works

```
Photo → SCRFD detector → Biometric guards → Umeyama align → ArcFace → embedding
```

**Biometric guards (auto-reject if failed):**

| Guard | Threshold | Rejects |
|---|---|---|
| Face size | ≥ 80 px | Too small / too far |
| Face centrality | 0.25 – 0.75 | Off-center faces |
| Yaw | 0.5 – 2.0 | Head turned sideways |
| Pitch | 0.4 – 1.8 | Head tilted up/down |
| CR-FIQA quality | ≥ 0.15 | Blurry / poor lighting |
| Multi-face | max 1 | More than one person |

---

## Installation

```bash
pip install onnxruntime opencv-python-headless numpy scikit-image
```

Place ONNX models in a `models/` directory next to `face_embed.py`.  
Models are **auto-detected by filename** — no configuration needed:

```
models/
├── scrfd_10g_bnkps.onnx          # face detector  (SCRFD family)
├── w600k_mbf.onnx                # face recognizer (any InsightFace ArcFace model)
└── crfiqa_s_quality_opset11.onnx # quality scorer  (optional)
```

**Supported recognizer models** (any InsightFace ArcFace variant):
`w600k_mbf`, `w600k_r50`, `w600k_r100`, `buffalo_sc`, `buffalo_s`, `buffalo_m`, `buffalo_l`, `antelopev2`

---

## Usage

```python
from face_embed import extract_embedding

# from file path
result = extract_embedding("photo.jpg")

# from raw bytes (e.g. HTTP upload)
result = extract_embedding(request.body)

if result["ok"]:
    embedding = result["embedding"]   # list[float], 512 values, L2-normalized
    quality   = result["quality"]     # float, 0.0 – 1.0  (CR-FIQA score)
else:
    error = result["error"]           # str, human-readable reason
```

**Return value:**

```python
# Success:
{
    "ok":        True,
    "embedding": [0.021, -0.043, ...],  # 512 floats, L2-normalized
    "quality":   0.724                  # CR-FIQA biometric quality score
}

# Failure:
{
    "ok":    False,
    "error": "Face turned too far sideways (yaw ratio=0.38), please look directly at camera"
}
```

---

## Exit Codes / Error Handling

The function never raises exceptions — all errors are returned as `{"ok": False, "error": "..."}`.

| Scenario | `ok` | `error` |
|---|---|---|
| Success | `True` | — |
| File not found / corrupted | `False` | `"Cannot read image ..."` |
| No face detected | `False` | `"No face detected in image"` |
| Multiple faces | `False` | `"Multiple faces detected (N) ..."` |
| Face too small | `False` | `"Face too small (WxHpx) ..."` |
| Face off-center | `False` | `"Face not centered (cx=...) ..."` |
| Head rotated | `False` | `"Face turned too far sideways ..."` |
| Head tilted | `False` | `"Face tilted too far up/down ..."` |
| Low quality | `False` | `"Face quality too low (score=...) ..."` |
| Model not found | `False` | `"'scrfd' model not found in: ..."` |

---

## File Structure

```
faceProject/
├── face_embed.py     ← the only file you need
├── requirements.txt
├── models/
│   ├── scrfd_10g_bnkps.onnx
│   ├── w600k_mbf.onnx
│   └── crfiqa_s_quality_opset11.onnx
└── legacy/           ← archived (FastAPI server, pipeline, database, debug tools)
```
