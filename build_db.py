#!/usr/bin/env python3
"""
build_db.py — Rasmlardan yuz embeddinglarini ajratib employee.db yaratadi.

  Rejim 1 — bitta kishi (web backend har upload da chaqiradi):
    python build_db.py --name "Ali Karimov" --photos a.jpg b.jpg --db employee.db

  Rejim 2 — papkadan to'liq bazani qayta qurish (batch):
    python build_db.py --photos-dir ./photos/ --db employee.db

Rasm tuzilmasi (Rejim 2):
    photos/
    ├── Ali_Karimov/
    │   ├── 1.jpg
    │   └── 2.jpg
    └── Vali_Toshmatov/
        └── front.jpg

Papka nomi xodim ismi sifatida ishlatiladi (Ali_Karimov → "Ali Karimov").
"""

import argparse
import glob
import os
import sys

import cv2

from database import EmployeeDB
from pipeline import FacePipeline

# ── Modellar joyi ─────────────────────────────────────────────────────────────
# Skript yonidagi models/ papkasidan avtomatik topiladi
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ── Model avtomatik topish ────────────────────────────────────────────────────

def auto_find_models(models_dir: str) -> dict:
    """
    models/ papkasidagi ONNX fayllarni fayl nomi bo'yicha avtomatik topadi.
    Qaytaradi: {"scrfd": "...", "arcface": "...", "crfiqa": "..."}
    """
    found = {}
    onnx_files = sorted(glob.glob(os.path.join(models_dir, "*.onnx")))
    for f in onnx_files:
        n = os.path.basename(f).lower()
        if "scrfd" in n and "scrfd" not in found:
            found["scrfd"] = f
        elif any(k in n for k in ("mbf", "w600k", "arcface", "r50", "r100")) and "arcface" not in found:
            found["arcface"] = f
        elif any(k in n for k in ("crfiqa", "quality", "fiqa")) and "crfiqa" not in found:
            found["crfiqa"] = f
    return found


def load_pipeline(models: dict) -> FacePipeline:
    """Modellarni tekshirib FacePipeline yuklaydi."""
    for required in ("scrfd", "arcface"):
        if required not in models:
            print(f"[ERROR] '{required}' modeli topilmadi: {MODELS_DIR}/")
            print(f"        Kutilayotgan fayl nomlari: scrfd*.onnx, w600k*.onnx yoki arcface*.onnx")
            sys.exit(1)

    print(f"[MODEL] SCRFD:   {models['scrfd']}")
    print(f"[MODEL] ArcFace: {models['arcface']}")
    if "crfiqa" in models:
        print(f"[MODEL] CR-FIQA: {models['crfiqa']}")
    else:
        print("[MODEL] CR-FIQA: topilmadi — sifat tekshiruvi o'chiriladi")

    return FacePipeline(
        scrfd_model_path=models["scrfd"],
        arcface_model_path=models["arcface"],
        crfiqa_model_path=models.get("crfiqa"),  # None bo'lsa ham ishlaydi
    )


# ── Bitta rasmni qayta ishlash ────────────────────────────────────────────────

def process_photo(pipeline: FacePipeline, photo_path: str) -> tuple:
    """
    Bitta rasmdan embedding ajratadi.

    Qaytaradi: (embedding: np.ndarray, quality_score: float)
    Xato bo'lsa: ValueError raise qiladi
    """
    img = cv2.imread(photo_path)
    if img is None:
        raise ValueError("Rasmni o'qib bo'lmadi (noto'g'ri format yoki fayl buzilgan)")

    # pipeline.process_image ValueError raise qiladi agar:
    # - yuz topilmasa
    # - bir nechta yuz bo'lsa
    # - yuz kichik bo'lsa (size guard)
    # - yuz markazda bo'lmasa (centrality guard)
    # - bosh burilgan bo'lsa (pose guard)
    # - sifat past bo'lsa (cr-fiqa guard)
    results = pipeline.process_image(img, max_faces=1)
    embed = results[0]["embed"]
    q_score = results[0]["quality_score"]
    return embed, q_score


# ── Bir kishini DB ga ro'yxatdan o'tkazish ───────────────────────────────────

def enroll_person(pipeline: FacePipeline, db: EmployeeDB,
                  name: str, photo_paths: list) -> tuple:
    """
    Bir kishining rasmlarini qayta ishlaydi va DB ga saqlaydi.

    Qaytaradi: (ok_count: int, failed: list of (path, reason))
    """
    embeddings = []
    failed = []

    for photo_path in photo_paths:
        if not os.path.isfile(photo_path):
            msg = "fayl mavjud emas"
            print(f"  [WARN] {photo_path}: {msg}")
            failed.append((photo_path, msg))
            continue

        try:
            embed, q_score = process_photo(pipeline, photo_path)
            embeddings.append(embed)
            print(f"  [OK]   {photo_path}  (quality={q_score:.3f})")

        except ValueError as e:
            print(f"  [WARN] {photo_path}: {e}")
            failed.append((photo_path, str(e)))

        except Exception as e:
            msg = f"kutilmagan xato: {e}"
            print(f"  [ERR]  {photo_path}: {msg}")
            failed.append((photo_path, msg))

    # Muvaffaqiyatli embeddinglar bo'lsagina DB ga yozish
    if embeddings:
        person_id = db.add_person(name)
        for emb in embeddings:
            db.add_embedding(person_id, emb)

    return len(embeddings), failed


# ── CLI argumentlar ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasmlardan yuz embeddinglarini ajratib employee.db yaratadi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Misollar:
  # Rejim 1 — bitta kishi (web backend har upload da chaqiradi):
  python build_db.py --name "Ali Karimov" --photos ali1.jpg ali2.jpg --db /srv/nfs/client_001.db

  # Rejim 2 — papkadan to'liq bazani qayta qurish (batch):
  python build_db.py --photos-dir ./photos/ --db /srv/nfs/client_001.db
        """,
    )

    parser.add_argument(
        "--db", required=True, metavar="PATH",
        help="Chiqish: employee.db fayl yo'li (mavjud bo'lmasa yaratiladi)",
    )

    # Rejim 1: bitta kishi
    parser.add_argument(
        "--name", metavar="NAME",
        help="Rejim 1: Xodim ismi, masalan: 'Ali Karimov'",
    )
    parser.add_argument(
        "--photos", nargs="+", metavar="FILE",
        help="Rejim 1: Rasm fayllari (bir yoki bir nechta)",
    )

    # Rejim 2: papkadan batch
    parser.add_argument(
        "--photos-dir", metavar="DIR",
        help="Rejim 2: photos/<Ism_Familiya>/*.jpg tuzilmasidagi papka",
    )

    return parser.parse_args()


# ── Asosiy mantiq ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Modellarni topib pipeline ni yuklash
    models = auto_find_models(MODELS_DIR)
    pipeline = load_pipeline(models)
    print()

    # ── Rejim 1: bitta kishi ──────────────────────────────────────────────────
    if args.name and args.photos:
        db = EmployeeDB(args.db)

        print(f"[INFO] Kishi   : '{args.name}'")
        print(f"[INFO] Rasmlar : {len(args.photos)} ta")
        print(f"[INFO] DB      : {args.db}")
        print()

        ok_count, failed = enroll_person(pipeline, db, args.name, args.photos)

        print()
        st = db.stats()
        if ok_count > 0:
            print(f"[DONE] '{args.name}': {ok_count} embedding saqlandi"
                  + (f", {len(failed)} rasm rad etildi" if failed else ""))
            print(f"[DB]   Jami: {st['persons']} kishi, "
                  f"{st['embeddings']} embedding, {st['size_kb']} KB")
        else:
            print(f"[FAIL] '{args.name}': hech qanday embedding saqlanmadi "
                  f"({len(failed)} rasm rad etildi)")
            sys.exit(1)

    # ── Rejim 2: papkadan to'liq qayta qurish ────────────────────────────────
    elif args.photos_dir:
        photos_dir = args.photos_dir.rstrip(os.sep)

        if not os.path.isdir(photos_dir):
            print(f"[ERROR] Papka topilmadi: {photos_dir}")
            sys.exit(1)

        # Eski DB ni o'chirib yangi toza DB yaratish
        if os.path.exists(args.db):
            os.remove(args.db)
            print(f"[INFO] Eski DB o'chirildi: {args.db}")

        db = EmployeeDB(args.db)

        # Papkadagi barcha kishi papkalarini topish
        person_dirs = sorted(
            d for d in os.listdir(photos_dir)
            if os.path.isdir(os.path.join(photos_dir, d))
        )

        if not person_dirs:
            print(f"[ERROR] {photos_dir}/ papkasida hech qanday kishi papkasi topilmadi")
            print(f"        Kutilayotgan tuzilma: {photos_dir}/<Ism_Familiya>/*.jpg")
            sys.exit(1)

        print(f"[INFO] {len(person_dirs)} ta kishi papkasi topildi")
        print(f"[INFO] DB: {args.db}")
        print()

        total_ok = 0
        total_fail = 0
        persons_enrolled = 0

        for person_dir in person_dirs:
            # Papka nomi → ism: "Ali_Karimov" → "Ali Karimov"
            name = person_dir.replace("_", " ")
            person_path = os.path.join(photos_dir, person_dir)

            photos = sorted(
                glob.glob(os.path.join(person_path, "*.jpg"))
                + glob.glob(os.path.join(person_path, "*.jpeg"))
                + glob.glob(os.path.join(person_path, "*.png"))
            )

            if not photos:
                print(f"[SKIP] {name}: rasm topilmadi")
                continue

            print(f"[INFO] {name} ({len(photos)} ta rasm):")
            ok_count, failed = enroll_person(pipeline, db, name, photos)
            total_ok += ok_count
            total_fail += len(failed)
            if ok_count > 0:
                persons_enrolled += 1
            print()

        # Yakuniy natija
        st = db.stats()
        print("─" * 52)
        print(f"[DONE]  Tugadi!")
        print(f"        {persons_enrolled}/{len(person_dirs)} kishi ro'yxatga olindi")
        print(f"        {total_ok} embedding saqlandi, {total_fail} rasm rad etildi")
        print(f"[DB]    {args.db}  ({st['size_kb']} KB)")

        if persons_enrolled == 0:
            print("[FAIL]  Hech qanday kishi saqlanmadi!")
            sys.exit(1)

    # ── Noto'g'ri argument kombinatsiyasi ─────────────────────────────────────
    else:
        print("[ERROR] Noto'g'ri argumentlar!\n")
        print("Rejim 1 — bitta kishi:")
        print("  python build_db.py --name 'Ali Karimov' --photos a.jpg b.jpg --db out.db\n")
        print("Rejim 2 — papkadan batch:")
        print("  python build_db.py --photos-dir ./photos/ --db out.db")
        sys.exit(1)


if __name__ == "__main__":
    main()
