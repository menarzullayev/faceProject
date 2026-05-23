#!/usr/bin/env python3
"""
Full Pipeline Visual Debugger (10 Steps)
Allows developers to run any face photo through the pipeline and inspect 
exactly how each guard, detecor, aligner, quality filter and vectorization step behaves.
"""
import os
import sys
import argparse
import cv2
import numpy as np

# Suppress ONNX runtime warnings
os.environ["ORT_LOGGING_LEVEL"] = "3"

from pipeline import FacePipeline

# Setup default models matching local environment dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRFD = os.path.join(BASE_DIR, "models", "scrfd_10g_bnkps.onnx")
DEFAULT_ARCFACE = os.path.join(BASE_DIR, "models", "w600k_mbf.onnx")
DEFAULT_CRFIQA = os.path.join(BASE_DIR, "models", "crfiqa_s_quality_opset11.onnx")
DEFAULT_TEST_IMAGE = "/srv/nfs/data/test_images/face_clear.jpg"
DEFAULT_DEBUG_DIR = os.path.join(BASE_DIR, "debug_output")

# Terminal Colors
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

def print_step(num: int, name: str, desc: str, status: str = "DONE"):
    col = C_GREEN if status == "DONE" else (C_YELLOW if status == "WARNING" else C_RED)
    print(f"{C_BOLD}[Step {num:02d}/10]{C_RESET} {C_CYAN}{name:<25}{C_RESET} | {desc:<55} | {col}[{status}]{C_RESET}")

def main():
    parser = argparse.ArgumentParser(description="Visual Pipeline Debugger (10 Steps)")
    parser.add_argument("--image", default=DEFAULT_TEST_IMAGE, help="Path to input BGR image")
    parser.add_argument("--out-dir", default=DEFAULT_DEBUG_DIR, help="Directory to save 10 debug images")
    parser.add_argument("--scrfd", default=DEFAULT_SCRFD, help="Path to SCRFD ONNX detector")
    parser.add_argument("--arcface", default=DEFAULT_ARCFACE, help="Path to ArcFace ONNX recognition model")
    parser.add_argument("--crfiqa", default=DEFAULT_CRFIQA, help="Path to CR-FIQA ONNX quality model")
    args = parser.parse_args()

    print("=" * 100)
    print(f"{C_BOLD}{C_BLUE}             🚀 AMBARELLA-COMPATIBLE FACE PIPELINE 10-STEP VISUAL DEBUGGER{C_RESET}")
    print("=" * 100)
    
    print(f"{C_BOLD}Configuration:{C_RESET}")
    print(f"  - Input Image:  {args.image}")
    print(f"  - Output Dir:   {args.out_dir}")
    print(f"  - SCRFD 10G:    {args.scrfd}")
    print(f"  - ArcFace MBF:  {args.arcface}")
    print(f"  - CR-FIQA Model: {args.crfiqa}")
    print("-" * 100)

    # 1. Check file existence
    if not os.path.exists(args.image):
        print(f"{C_RED}Error: Input image '{args.image}' not found!{C_RESET}")
        sys.exit(1)
    
    # 2. Initialize pipeline
    print(f"{C_YELLOW}Initializing models inside FacePipeline...{C_RESET}")
    try:
        pipeline = FacePipeline(args.scrfd, args.arcface, args.crfiqa)
    except Exception as e:
        print(f"{C_RED}Failed to load ONNX sessions: {e}{C_RESET}")
        sys.exit(1)

    # 3. Read image
    img = cv2.imread(args.image)
    if img is None:
        print(f"{C_RED}Error: OpenCV failed to read '{args.image}'!{C_RESET}")
        sys.exit(1)

    print(f"{C_GREEN}Initialization complete. Executing 10 Debug Stages...{C_RESET}\n")

    try:
        # We run our beautifully annotated debugging function
        results = pipeline.process_image_debug(img, args.out_dir)
        res = results[0]

        # Log console steps
        print_step(1, "Raw Input Image", f"Saved raw BGR photo as 01_raw_input.jpg")
        print_step(2, "Letterbox Resize", f"Padded/resized image to 640x640 (02_letterbox_input.jpg)")
        print_step(3, "Raw Detection", f"Extracted all bounding box candidates (03_detection_raw.jpg)")
        print_step(4, "Filtered Detection", f"Kept best face (score: {res['score']:.4f}) with keypoints (04_detection_filtered.jpg)")
        
        w, h = int(res['bbox'][2] - res['bbox'][0]), int(res['bbox'][3] - res['bbox'][1])
        print_step(5, "Face Size Guard", f"Verified bounding box size: {w}x{h} px >= {pipeline.min_face_size}px (05_size_check.jpg)")
        
        W, H = img.shape[1], img.shape[0]
        cx, cy = (res['bbox'][0] + res['bbox'][2]) / 2.0, (res['bbox'][1] + res['bbox'][3]) / 2.0
        print_step(6, "Face Centrality Guard", f"Verified center ratios: ({cx/W:.2f}, {cy/H:.2f}) inside central region (06_centrality_check.jpg)")
        
        print_step(7, "Yaw & Pitch Pose Guard", f"Verified face asymmetry levels within Yaw/Pitch bounds (07_pose_check.jpg)")
        print_step(8, "Umeyama Alignment", f"Aligned and cropped face to standard 112x112 layout (08_umeyama_aligned.jpg)")
        print_step(9, "CR-FIQA Quality Guard", f"Biometric Quality Score evaluated: {res['quality_score']:.4f} >= {pipeline.min_quality_score} (09_crfiqa_quality.jpg)")
        
        print_step(10, "ArcFace L2 Normalization", f"Extracted L2-normalized 512-dim embedding with vector preview panel (10_final_arcface.jpg)")

        print("\n" + "=" * 100)
        print(f"{C_GREEN}{C_BOLD}✅ SUCCESS: Photo has passed ALL 10 validation and embedding extraction stages!{C_RESET}")
        print(f"Debug images written to: {C_YELLOW}{args.out_dir}/{C_RESET}")
        print(f"First 5 dimensions of normalized embedding: {res['embed'][:5]}")
        print("=" * 100)

    except Exception as e:
        print(f"\n{C_RED}{C_BOLD}❌ REJECTED: Pipeline failed validation check!{C_RESET}")
        print(f"{C_RED}Error details: {e}{C_RESET}")
        print(f"Partial debug images (up to failure step) saved in: {C_YELLOW}{args.out_dir}/{C_RESET}")
        print("=" * 100)
        sys.exit(1)

if __name__ == "__main__":
    main()
