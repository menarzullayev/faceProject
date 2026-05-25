"""
SQLite employee database — ultra-lightweight schema matching camera's employee.db exactly.
Optimized for Ambarella CV25 NPU read performance.
"""
import sqlite3
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

EMBED_DIM = 512
EMBED_BYTES = EMBED_DIM * 4  # float32


def _pack_embed(embed: np.ndarray) -> bytes:
    """512 float32 → 2048 bytes (raw, no header)."""
    return embed.astype(np.float32).tobytes()


def _unpack_embed(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class EmployeeDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        # Maximize NPU read efficiency: standard DELETE journal mode without extra files
        conn.execute("PRAGMA journal_mode = DELETE;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS person (
                    id         INTEGER PRIMARY KEY,
                    name       TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id  INTEGER NOT NULL REFERENCES person(id),
                    embed      BLOB    NOT NULL
                );
                -- Speed up face query lookups on NPU
                CREATE INDEX IF NOT EXISTS idx_embedding_person ON embedding(person_id);
            """)

    def vacuum(self):
        """Minimize database size on SD card to maximize NPU boot speed."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM;")
            conn.close()
        except Exception:
            pass

    # ── Person CRUD ───────────────────────────────────────────────────────────

    def add_person(self, name: str) -> int:
        """Insert a new person. Returns person_id."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO person(name) VALUES (?)",
                (name,)
            )
            pid = cur.lastrowid
        self.vacuum()
        return pid

    def get_person(self, person_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM person WHERE id=?", (person_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_persons(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT p.*, COUNT(e.id) AS embed_count "
                "FROM person p LEFT JOIN embedding e ON e.person_id=p.id "
                "GROUP BY p.id ORDER BY p.id"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_person(self, person_id: int) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM embedding WHERE person_id=?", (person_id,))
            cur = conn.execute("DELETE FROM person WHERE id=?", (person_id,))
            rowcount = cur.rowcount
        self.vacuum()
        return rowcount > 0

    # ── Embedding CRUD ────────────────────────────────────────────────────────

    def add_embedding(self, person_id: int, embed: np.ndarray) -> int:
        """Store L2-normalized embedding. Returns embedding id."""
        norm = np.linalg.norm(embed)
        normalized = (embed / norm).astype(np.float32) if norm > 1e-9 else embed
        blob = _pack_embed(normalized)
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO embedding(person_id, embed) VALUES (?,?)",
                (person_id, blob)
            )
            eid = cur.lastrowid
        self.vacuum()
        return eid

    def get_embeddings(self, person_id: int) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, embed FROM embedding WHERE person_id=?",
                (person_id,)
            ).fetchall()
            return [{"id": r["id"], "embed": _unpack_embed(r["embed"])} for r in rows]

    def delete_embedding(self, embedding_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM embedding WHERE id=?", (embedding_id,))
            rowcount = cur.rowcount
        self.vacuum()
        return rowcount > 0

    def get_all_embeddings(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT e.person_id, p.name, e.embed "
                "FROM embedding e JOIN person p ON e.person_id=p.id"
            ).fetchall()
            return [{"person_id": r["person_id"], "name": r["name"],
                     "embed": _unpack_embed(r["embed"])} for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as conn:
            persons = conn.execute("SELECT COUNT(*) FROM person").fetchone()[0]
            embeds = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0]
            size_kb = Path(self.db_path).stat().st_size // 1024 if Path(self.db_path).exists() else 0
            return {"persons": persons, "embeddings": embeds, "size_kb": size_kb}
