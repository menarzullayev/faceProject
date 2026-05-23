"""
InsightFace enrollment pipeline.
Detection (SCRFD) + Alignment (Umeyama) + Embedding (ArcFace).
Enhanced with:
  - 📏 Face Size Filter
  - 🧭 Yaw & Pitch Pose Limits
  - 🎯 Face Centrality
  - 🏆 CR-FIQA Integration
"""
import sys
import os
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from skimage.transform import SimilarityTransform

# ── ArcFace standard 5-point template (112×112) ──────────────────────────────
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

SCRFD_INPUT_SIZE = 640
ARCFACE_INPUT_SIZE = 112
DETECTION_THRESHOLD = 0.30
NMS_THRESHOLD = 0.40


class FacePipeline:
    """
    Full face enrollment pipeline:
      image → detect → align → validate (Size, Centrality, Pose, CR-FIQA) → embed
    """

    def __init__(self, scrfd_model_path: str, arcface_model_path: str, crfiqa_model_path: str = None):
        providers = ["CPUExecutionProvider"]
        self.scrfd = ort.InferenceSession(scrfd_model_path, providers=providers)
        self.arcface = ort.InferenceSession(arcface_model_path, providers=providers)
        self._scrfd_input_name = self.scrfd.get_inputs()[0].name
        self._arc_input_name = self.arcface.get_inputs()[0].name

        # Load optional CR-FIQA Quality Assessment model
        self.crfiqa = None
        if crfiqa_model_path and os.path.exists(crfiqa_model_path):
            print(f"[pipeline] Loading CR-FIQA: {crfiqa_model_path}")
            self.crfiqa = ort.InferenceSession(crfiqa_model_path, providers=providers)
            self._crfiqa_input_name = self.crfiqa.get_inputs()[0].name
        else:
            print("[pipeline] CR-FIQA path not provided or file not found, running without FIQA check.")

        # ── Dynamic Configuration (Limits) ──
        self.min_face_size = 80             # 📏 Bbox width/height min pixels
        self.min_cx_ratio = 0.25            # 🎯 Face center x ratio min
        self.max_cx_ratio = 0.75            # 🎯 Face center x ratio max
        self.min_cy_ratio = 0.25            # 🎯 Face center y ratio min
        self.max_cy_ratio = 0.75            # 🎯 Face center y ratio max
        self.min_yaw_ratio = 0.5            # 🧭 Min eye-to-nose symmetry ratio (Yaw)
        self.max_yaw_ratio = 2.0            # 🧭 Max eye-to-nose symmetry ratio (Yaw)
        self.min_pitch_ratio = 0.4          # 🧭 Min eye-to-mouth symmetry ratio (Pitch)
        self.max_pitch_ratio = 1.8          # 🧭 Max eye-to-mouth symmetry ratio (Pitch)
        self.min_quality_score = 0.15       # 🏆 CR-FIQA min score threshold
        self.similarity_threshold = 0.35    # 🎚️ Cosine similarity threshold for matching

    # ── Detection ─────────────────────────────────────────────────────────────

    def _letterbox(self, img: np.ndarray, size: int = SCRFD_INPUT_SIZE):
        """Official InsightFace SCRFD Top-Left padding implementation."""
        h, w = img.shape[:2]
        im_ratio = float(h) / w
        model_ratio = 1.0  # since size x size is square 640x640, ratio is 1.0
        
        if im_ratio > model_ratio:
            new_height = size
            new_width = int(new_height / im_ratio)
        else:
            new_width = size
            new_height = int(new_width * im_ratio)
            
        scale = float(new_height) / h
        resized = cv2.resize(img, (new_width, new_height))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_height, :new_width, :] = resized
        
        # In official top-left padding, padding offset is 0
        return canvas, scale, 0, 0

    def detect(self, img_bgr: np.ndarray, threshold: float = DETECTION_THRESHOLD):
        """
        Run SCRFD detection.
        Returns list of dicts: {bbox, kps5, score}
        """
        h, w = img_bgr.shape[:2]
        lb, scale, px, py = self._letterbox(img_bgr, SCRFD_INPUT_SIZE)

        blob = cv2.dnn.blobFromImage(
            lb, 1.0 / 128.0, (SCRFD_INPUT_SIZE, SCRFD_INPUT_SIZE),
            (127.5, 127.5, 127.5), swapRB=True
        )
        outs = self.scrfd.run(None, {self._scrfd_input_name: blob})

        strides = [8, 16, 32]
        feat_sizes = [SCRFD_INPUT_SIZE // s for s in strides]
        results = []

        for si, stride in enumerate(strides):
            fs = feat_sizes[si]
            scores = outs[si].reshape(-1)
            bboxes = outs[3 + si].reshape(-1, 4)
            kpss = outs[6 + si].reshape(-1, 5, 2)

            cx = np.tile(np.repeat(np.arange(fs), 2), fs) * stride
            cy = np.repeat(np.arange(fs) * stride, fs * 2)

            for i in range(len(scores)):
                sc = float(scores[i])
                if sc < threshold:
                    continue
                dx, dy, dw, dh = bboxes[i]
                x1 = (cx[i] - dx * stride - px) / scale
                y1 = (cy[i] - dy * stride - py) / scale
                x2 = (cx[i] + dw * stride - px) / scale
                y2 = (cy[i] + dh * stride - py) / scale
                kps = np.array([
                    [(cx[i] + kp[0] * stride - px) / scale,
                     (cy[i] + kp[1] * stride - py) / scale]
                    for kp in kpss[i]
                ], dtype=np.float32)
                results.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "kps5": kps,
                    "score": sc,
                })

        if len(results) > 1:
            boxes = [[r["bbox"][0], r["bbox"][1],
                      r["bbox"][2] - r["bbox"][0],
                      r["bbox"][3] - r["bbox"][1]] for r in results]
            scores = [r["score"] for r in results]
            idx = cv2.dnn.NMSBoxes(boxes, scores, threshold, NMS_THRESHOLD)
            results = [results[i] for i in idx.flatten()]

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ── Alignment ─────────────────────────────────────────────────────────────

    def align(self, img_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
        """Umeyama similarity transform → 112×112 BGR aligned face."""
        tform = SimilarityTransform()
        tform.estimate(kps5, ARCFACE_DST)
        M = tform.params[:2]
        return cv2.warpAffine(img_bgr, M, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
                               flags=cv2.INTER_LINEAR)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        ArcFace inference on 112×112 BGR face.
        Returns L2-normalized 512-dim float32 vector.
        """
        blob = cv2.dnn.blobFromImage(
            face_bgr, 1.0 / 128.0, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
            (127.5, 127.5, 127.5), swapRB=True
        )
        raw = self.arcface.run(None, {self._arc_input_name: blob})[0][0]
        norm = np.linalg.norm(raw)
        return (raw / norm).astype(np.float32) if norm > 1e-9 else raw.astype(np.float32)

    # ── Full pipeline ────────────────────────────────────────────────_________

    def process_image(self, img_bgr: np.ndarray, max_faces: int = 1):
        """
        Detect + validate (Size, Centrality, Pose, CR-FIQA) + align + embed.
        Returns list of {embed, kps5, score, face_img} for up to max_faces.
        Raises ValueError if any validation check fails or no face detected.
        """
        H, W = img_bgr.shape[:2]
        faces = self.detect(img_bgr)
        if not faces:
            raise ValueError("No face detected in image")
        if len(faces) > 1:
            raise ValueError(f"Multiple faces detected ({len(faces)}), only 1 face allowed per photo")

        results = []
        for face in faces[:max_faces]:
            bbox = face["bbox"]
            kps5 = face["kps5"]

            # 📏 1. Face Size Filter
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_face_size or h < self.min_face_size:
                raise ValueError(
                    f"Face is too small ({int(w)}x{int(h)}), minimum allowed size is {self.min_face_size}x{self.min_face_size} pixels."
                )

            # 🎯 2. Face Centrality
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            cx_ratio = cx / W
            cy_ratio = cy / H
            if not (self.min_cx_ratio <= cx_ratio <= self.max_cx_ratio) or not (self.min_cy_ratio <= cy_ratio <= self.max_cy_ratio):
                raise ValueError(
                    f"Face is not centered (cx={cx_ratio:.2f}, cy={cy_ratio:.2f}). Bounding box center must be within central area."
                )

            # 🧭 3. Yaw & Pitch Pose Limits
            # Yaw (Left eye-to-nose vs Right eye-to-nose)
            d_left = np.linalg.norm(kps5[2] - kps5[0])
            d_right = np.linalg.norm(kps5[2] - kps5[1])
            yaw_ratio = max(d_left, 1e-5) / max(d_right, 1e-5)
            if yaw_ratio < self.min_yaw_ratio or yaw_ratio > self.max_yaw_ratio:
                raise ValueError(
                    f"Face is turned too far sideways (Yaw symmetry ratio: {yaw_ratio:.2f}). Please look directly at the camera."
                )

            # Pitch (Midpoint of eyes-to-nose vs nose-to-midpoint of mouth)
            eye_mid = (kps5[0] + kps5[1]) / 2.0
            mouth_mid = (kps5[3] + kps5[4]) / 2.0
            d_eye_nose = np.linalg.norm(kps5[2] - eye_mid)
            d_nose_mouth = np.linalg.norm(kps5[2] - mouth_mid)
            pitch_ratio = max(d_eye_nose, 1e-5) / max(d_nose_mouth, 1e-5)
            if pitch_ratio < self.min_pitch_ratio or pitch_ratio > self.max_pitch_ratio:
                raise ValueError(
                    f"Face is tilted too far up or down (Pitch symmetry ratio: {pitch_ratio:.2f}). Please keep your face level."
                )

            # Align face crop (112x112)
            aligned = self.align(img_bgr, kps5)

            # 🏆 4. CR-FIQA Integration
            q_score = 1.0  # Default value if CR-FIQA session not loaded
            if self.crfiqa is not None:
                # Normalization baked inside the model
                blob_q = cv2.dnn.blobFromImage(
                    aligned, 1.0 / 128.0, (112, 112),
                    (127.5, 127.5, 127.5), swapRB=True
                )
                q_score = float(self.crfiqa.run(None, {self._crfiqa_input_name: blob_q})[0][0][0])
                if q_score < self.min_quality_score:
                    raise ValueError(
                        f"Face biometric quality score is too low ({q_score:.4f}), minimum required is {self.min_quality_score:.4f}. Please use a higher quality photo."
                    )

            embedding = self.embed(aligned)
            results.append({
                "embed": embedding,
                "kps5": kps5,
                "score": face["score"],
                "bbox": face["bbox"],
                "face_img": aligned,
                "quality_score": q_score,
            })
        return results

    def process_image_debug(self, img_bgr: np.ndarray, debug_dir: str):
        """
        Runs the full pipeline while saving 10 distinct visual debug steps
        to debug_dir. Creates the directory if it does not exist.
        """
        import os
        os.makedirs(debug_dir, exist_ok=True)
        H, W = img_bgr.shape[:2]
        
        # 1. Raw Input
        cv2.imwrite(os.path.join(debug_dir, "01_raw_input.jpg"), img_bgr)
        
        # 2. Letterbox Input
        lb, scale, px, py = self._letterbox(img_bgr, SCRFD_INPUT_SIZE)
        cv2.imwrite(os.path.join(debug_dir, "02_letterbox_input.jpg"), lb)
        
        # 3. Detection Raw
        faces_all = self.detect(img_bgr, threshold=0.1)  # Low threshold to show all candidates
        img_step3 = img_bgr.copy()
        for idx, face in enumerate(faces_all):
            bbox = face["bbox"]
            cv2.rectangle(img_step3, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img_step3, f"#{idx} {face['score']:.2f}", (int(bbox[0]), int(bbox[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, "03_detection_raw.jpg"), img_step3)
        
        # Filter down for NMS and max_faces = 1
        faces = self.detect(img_bgr)
        if not faces:
            img_failed = img_bgr.copy()
            cv2.putText(img_failed, "FAILED: No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "04_detection_filtered.jpg"), img_failed)
            raise ValueError("No face detected in image")
            
        face = faces[0]
        bbox = face["bbox"]
        kps5 = face["kps5"]
        
        # 4. Detection Filtered
        img_step4 = img_bgr.copy()
        cv2.rectangle(img_step4, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
        for kp in kps5:
            cv2.circle(img_step4, (int(kp[0]), int(kp[1])), 4, (255, 0, 0), -1)
        cv2.putText(img_step4, f"Selected (score: {face['score']:.4f})", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "04_detection_filtered.jpg"), img_step4)
        
        # 5. Size Check
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        img_step5 = img_bgr.copy()
        is_size_ok = (w >= self.min_face_size and h >= self.min_face_size)
        color_size = (0, 255, 0) if is_size_ok else (0, 0, 255)
        cv2.rectangle(img_step5, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_size, 3)
        cv2.putText(img_step5, f"Size: {int(w)}x{int(h)} (Min: {self.min_face_size}x{self.min_face_size})", 
                    (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_size, 2)
        cv2.imwrite(os.path.join(debug_dir, "05_size_check.jpg"), img_step5)
        if not is_size_ok:
            raise ValueError(f"Face is too small ({int(w)}x{int(h)}), minimum allowed size is {self.min_face_size}x{self.min_face_size} pixels.")
            
        # 6. Centrality Check
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        cx_ratio = cx / W
        cy_ratio = cy / H
        img_step6 = img_bgr.copy()
        
        # Draw face bounding box outline for reference (Light Gray)
        cv2.rectangle(img_step6, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (200, 200, 200), 1)
        
        # Draw target central region (Yellow)
        cv2.rectangle(img_step6, (int(W * self.min_cx_ratio), int(H * self.min_cy_ratio)),
                      (int(W * self.max_cx_ratio), int(H * self.max_cy_ratio)), (0, 255, 255), 2)
        
        is_centered = (self.min_cx_ratio <= cx_ratio <= self.max_cx_ratio) and (self.min_cy_ratio <= cy_ratio <= self.max_cy_ratio)
        color_center = (0, 255, 0) if is_centered else (0, 0, 255)
        cv2.circle(img_step6, (int(cx), int(cy)), 6, color_center, -1)
        cv2.putText(img_step6, f"Center: ({cx_ratio:.2f}, {cy_ratio:.2f})", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_center, 2)
        cv2.imwrite(os.path.join(debug_dir, "06_centrality_check.jpg"), img_step6)
        if not is_centered:
            raise ValueError(f"Face is not centered (cx={cx_ratio:.2f}, cy={cy_ratio:.2f}). Bounding box center must be within central area.")
            
        # 7. Pose Check
        d_left = np.linalg.norm(kps5[2] - kps5[0])
        d_right = np.linalg.norm(kps5[2] - kps5[1])
        yaw_ratio = max(d_left, 1e-5) / max(d_right, 1e-5)
        
        eye_mid = (kps5[0] + kps5[1]) / 2.0
        mouth_mid = (kps5[3] + kps5[4]) / 2.0
        d_eye_nose = np.linalg.norm(kps5[2] - eye_mid)
        d_nose_mouth = np.linalg.norm(kps5[2] - mouth_mid)
        pitch_ratio = max(d_eye_nose, 1e-5) / max(d_nose_mouth, 1e-5)
        
        img_step7 = img_bgr.copy()
        cv2.line(img_step7, (int(kps5[0][0]), int(kps5[0][1])), (int(kps5[2][0]), int(kps5[2][1])), (255, 255, 0), 1)
        cv2.line(img_step7, (int(kps5[1][0]), int(kps5[1][1])), (int(kps5[2][0]), int(kps5[2][1])), (255, 255, 0), 1)
        cv2.line(img_step7, (int(eye_mid[0]), int(eye_mid[1])), (int(kps5[2][0]), int(kps5[2][1])), (0, 255, 255), 1)
        cv2.line(img_step7, (int(mouth_mid[0]), int(mouth_mid[1])), (int(kps5[2][0]), int(kps5[2][1])), (0, 255, 255), 1)
        
        is_yaw_ok = (self.min_yaw_ratio <= yaw_ratio <= self.max_yaw_ratio)
        is_pitch_ok = (self.min_pitch_ratio <= pitch_ratio <= self.max_pitch_ratio)
        
        cv2.putText(img_step7, f"Yaw Ratio: {yaw_ratio:.2f} ({self.min_yaw_ratio}-{self.max_yaw_ratio})",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if is_yaw_ok else (0, 0, 255), 2)
        cv2.putText(img_step7, f"Pitch Ratio: {pitch_ratio:.2f} ({self.min_pitch_ratio}-{self.max_pitch_ratio})",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if is_pitch_ok else (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "07_pose_check.jpg"), img_step7)
        
        if not is_yaw_ok or not is_pitch_ok:
            raise ValueError(f"Pose check failed: Yaw={yaw_ratio:.2f}, Pitch={pitch_ratio:.2f}")
            
        # 8. Umeyama Aligned
        aligned = self.align(img_bgr, kps5)
        cv2.imwrite(os.path.join(debug_dir, "08_umeyama_aligned.jpg"), aligned)
        
        # 9. CR-FIQA Quality Check
        q_score = 1.0
        if self.crfiqa is not None:
            blob_q = cv2.dnn.blobFromImage(
                aligned, 1.0 / 128.0, (112, 112),
                (127.5, 127.5, 127.5), swapRB=True
            )
            q_score = float(self.crfiqa.run(None, {self._crfiqa_input_name: blob_q})[0][0][0])
            
        img_step9 = aligned.copy()
        is_q_ok = (q_score >= self.min_quality_score)
        color_q = (0, 255, 0) if is_q_ok else (0, 0, 255)
        cv2.putText(img_step9, f"Quality: {q_score:.4f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_q, 1)
        cv2.putText(img_step9, f"Min: {self.min_quality_score:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, "09_crfiqa_quality.jpg"), img_step9)
        
        if not is_q_ok:
            raise ValueError(f"Face biometric quality score is too low ({q_score:.4f}), minimum required is {self.min_quality_score:.4f}.")
            
        # 10. Final ArcFace Embedding Preview
        embedding = self.embed(aligned)
        
        # Generate an elegant visual graph of the embedding vector
        panel = np.ones((112, 200, 3), dtype=np.uint8) * 245
        cv2.putText(panel, "Embedding Vector", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(panel, "First 64 dims preview:", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        
        for i in range(min(64, len(embedding))):
            val = embedding[i]
            center_x = 100
            bar_len = int(val * 400)
            y_pos = 40 + i
            if y_pos >= 110:
                break
            color = (200, 50, 50) if val >= 0 else (50, 50, 200)
            cv2.line(panel, (center_x, y_pos), (center_x + bar_len, y_pos), color, 1)
            
        cv2.line(panel, (100, 38), (100, 110), (120, 120, 120), 1)
        img_step10 = np.hstack((aligned, panel))
        cv2.imwrite(os.path.join(debug_dir, "10_final_arcface.jpg"), img_step10)
        
        return [{
            "embed": embedding,
            "kps5": kps5,
            "score": face["score"],
            "bbox": face["bbox"],
            "face_img": aligned,
            "quality_score": q_score,
        }]

    def process_image_bytes(self, image_bytes: bytes, max_faces: int = 1):
        """Process image from raw bytes."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        return self.process_image(img, max_faces)
