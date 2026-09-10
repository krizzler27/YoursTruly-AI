"""Hybrid store for RAG chunks.

Two stores, one contract:
- LanceDB (backend/lancedb): vector rows, keyed by `{doc_id}:{index}`.
- SQLite FTS5 (same yourstrulyai.db): keyword index over the chunk text.
Both write/delete together so they never drift.
"""

from pathlib import Path
from typing import List, Optional
from uuid import UUID

import lancedb
from sqlalchemy import text
from sqlalchemy.orm import Session

from schemas.rag_schemas import Chunk

FTS_TABLE = "document_chunks_fts"


def ensure_fts_table(db: Session) -> None:
    """Create the FTS5 index once; later calls are a read-only no-op."""

    exists = db.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
        {"t": FTS_TABLE},
    ).first()

    if exists:
        return

    db.execute(
        text(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} "
            "USING fts5(id UNINDEXED, document_id UNINDEXED, text)"
        )
    )
    db.commit()


class LanceRepository:
    def __init__(self, db: Session, base_dir: Optional[Path] = None):
        self.db = db
        self.base_dir = base_dir or Path(__file__).parent.parent / "lancedb"
        self.conn = lancedb.connect(str(self.base_dir))
        ensure_fts_table(self.db)

    # ---- write path ----

    def upsert_chunks(
        self, document_id: UUID, chunks: List[Chunk], commit: bool = True
    ) -> None:
        """Replace a document's vectors + fts index (safe to re-run)."""
        self.delete_document(document_id, commit=False)
        did = str(document_id)
        rows = [
            {
                "id": f"{did}:{c.index}",
                "document_id": did,
                "index": c.index,
                "text": c.text,
                "vector": c.vector,
            }
            for c in chunks
        ]
        if rows:
            self._check_dim([r["vector"] for r in rows])
            table = self._table()
            if table is None:
                # first-ever table: schema inferred from the first real row
                self.conn.create_table("chunks", data=[rows[0]], mode="overwrite")
                rows = rows[1:]
                table = self._table()
            if rows:
                table.add(rows)
        self.db.execute(
            text(
                f"INSERT INTO {FTS_TABLE} (id, document_id, text) "
                "VALUES (:id, :did, :text)"
            ),
            [{"id": f"{did}:{c.index}", "did": did, "text": c.text} for c in chunks],
        )
        if commit:
            self.db.commit()

    def delete_document(self, document_id: UUID, commit: bool = True) -> None:
        self.delete_vectors(document_id)
        self.delete_fts(document_id)
        if commit:
            self.db.commit()

    def delete_fts(self, document_id: UUID) -> None:
        """FTS-only removal; joins the caller's transaction (no commit)."""
        self.db.execute(
            text(f"DELETE FROM {FTS_TABLE} WHERE document_id = :did"),
            {"did": str(document_id)},
        )

    def delete_vectors(self, document_id: UUID) -> None:
        """LanceDB-only removal (compensation path; no sqlite, no commit)."""
        table = self._table()
        if table is not None:
            table.delete(f"document_id = '{document_id}'")

    def _table(self):
        """Open the chunks table, or None when it doesn't exist yet."""
        try:
            if "chunks" in self.conn.table_names():
                return self.conn.open_table("chunks")
            return None
        except Exception:
            return None

    def _check_dim(self, vectors: List[list]) -> None:
        """Fail fast on dim drift (e.g. embed model swapped mid-corpus)."""
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            raise ValueError(f"mixed vector dims in one upsert: {sorted(dims)}")
        table = self._table()
        if table is not None:
            try:
                size = table.schema.field("vector").type.list_size
                if size is not None and dims and next(iter(dims)) != size:
                    raise ValueError(
                        f"vector dim {next(iter(dims))} != table dim {size}"
                    )
            except ValueError:
                raise
            except Exception:
                pass
