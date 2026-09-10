"""Hybrid store for RAG chunks.

Two stores, one contract:
- LanceDB (backend/lancedb): vector rows, keyed by `{doc_id}:{index}`.
- SQLite FTS5 (same yourstrulyai.db): keyword index over the chunk text.
Both write/delete together so they never drift.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

import lancedb
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import DocumentChunksModel
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
    def __init__(
        self,
        db: Session,
        base_dir: Optional[Path] = None,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self.db = db
        self.base_dir = base_dir or Path(__file__).parent.parent / "lancedb"
        self.conn = lancedb.connect(str(self.base_dir))
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

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

    # ---- read path (hybrid retrieval) ----

    def vector_search(self, query_vector: List[float], limit: int = 20) -> List[str]:
        """Vector top-K ids by L2 (unit-norm vectors rank as cosine)."""
        if not query_vector:
            return []

        table = self._table()

        if table is None:
            return []

        rows = table.search(query_vector).limit(limit).to_list()

        return [r["id"] for r in rows if "id" in r]

    def fts_search(self, query: str, limit: int = 20) -> List[str]:
        """BM25 top-K ids over chunk text (OR of word tokens)."""
        tokens = re.findall(r"\w+", query or "", flags=re.UNICODE)

        if not tokens:
            return []

        exists = self.db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
            {"t": FTS_TABLE},
        ).first()

        if not exists:
            return []

        match = " OR ".join(tokens)

        rows = self.db.execute(
            text(
                f"SELECT id FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :q "
                f"ORDER BY bm25({FTS_TABLE}) LIMIT :n"
            ),
            {"q": match, "n": limit},
        ).all()

        return [r[0] for r in rows]

    @staticmethod
    def rrf_merge(
        ranked_lists: List[List[str]], top_k: int = 5, rrf_k: int = 60
    ) -> List[str]:
        """Reciprocal rank fusion over id rankings."""
        scores: Dict[str, float] = {}

        for ranking in ranked_lists:
            for rank, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        return [cid for cid, _ in ordered[:top_k]]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Vector + BM25 candidates fused by RRF, hydrated from sqlite."""
        depth = candidate_k or self.candidate_k
        fusion_k = rrf_k if rrf_k is not None else self.rrf_k

        vector_ids = self.vector_search(query_vector, limit=depth)
        fts_ids = self.fts_search(query_text, limit=depth)

        if not vector_ids and not fts_ids:
            return []

        scores: Dict[str, float] = {}

        for ranking in (vector_ids, fts_ids):
            for rank, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (fusion_k + rank + 1)

        merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        ids = [cid for cid, _ in merged]

        rows = (
            self.db.query(DocumentChunksModel)
            .filter(DocumentChunksModel.id.in_(ids))
            .all()
        )

        by_id = {r.id: r for r in rows}

        hits: List[Dict[str, Any]] = []

        for cid, score in merged:
            row = by_id.get(cid)

            if row is None:
                continue

            hits.append(
                {
                    "id": row.id,
                    "document_id": str(row.document_id),
                    "index": row.index,
                    "heading": row.heading,
                    "page": row.page,
                    "text": row.text,
                    "score": score,
                }
            )

        return hits

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
