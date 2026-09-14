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


def sid(value: object) -> str:
    """Canonical string id for LanceDB/FTS/chunk keys: dashed UUID form.

    SQLite UUID columns store dashless hex via SQLAlchemy; every free-text
    store must use this helper so ids compare equal across stores.
    """
    return str(value)


def chunk_id(document_id: object, index: int) -> str:
    """Chunk PK shared by LanceDB rows, FTS rows, and SQLite chunks."""
    return f"{sid(document_id)}:{index}"


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
        self,
        document_id: UUID,
        chunks: List[Chunk],
        commit: bool = True,
        conversation_id: Optional[UUID] = None,
    ) -> None:
        """Replace a document's vectors + fts index (safe to re-run)."""
        self.delete_document(document_id, commit=False)

        did = sid(document_id)
        cid = sid(conversation_id) if conversation_id is not None else None

        rows = [
            {
                "id": chunk_id(did, c.index),
                "document_id": did,
                "conversation_id": cid,
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
                # first-ever table: schema inferred from the first real
                # row; use the returned handle, no list-then-reopen.
                table = self.conn.create_table(
                    "chunks", data=[rows[0]], mode="overwrite"
                )
                rows = rows[1:]

            if rows:
                table.add(rows)

        self.db.execute(
            text(
                f"INSERT INTO {FTS_TABLE} (id, document_id, text) "
                "VALUES (:id, :did, :text)"
            ),
            [{"id": chunk_id(did, c.index), "did": did, "text": c.text} for c in chunks],
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
            {"did": sid(document_id)},
        )

    def delete_vectors(self, document_id: UUID) -> None:
        """LanceDB-only removal (compensation path; no sqlite, no commit)."""
        table = self._table()

        if table is not None:
            table.delete(f"document_id = '{document_id}'")

    # ---- read path (hybrid retrieval) ----

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 20,
        conversation_id: Optional[UUID] = None,
    ) -> List[str]:
        """Vector top-K ids by L2 (unit-norm vectors rank as cosine)."""
        if not query_vector:
            return []

        table = self._table()

        if table is None:
            return []

        query = table.search(query_vector)

        if conversation_id is not None:
            # UUID charset is filter-safe; prefilter keeps ranking scoped.
            query = query.where(f"conversation_id = '{conversation_id}'")

        rows = query.limit(limit).to_list()

        return [r["id"] for r in rows if "id" in r]

    def fts_search(
        self, query: str, limit: int = 20, doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """BM25 top-K ids over chunk text (OR of quoted word tokens)."""
        tokens = re.findall(r"\w+", query or "", flags=re.UNICODE)

        if not tokens:
            return []

        if doc_ids is not None and not doc_ids:
            return []  # scoped chat with nothing indexed: no query needed

        exists = self.db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
            {"t": FTS_TABLE},
        ).first()

        if not exists:
            return []

        # Quote each token: bare OR/AND/NOT otherwise break FTS5 syntax.
        match = " OR ".join(f'"{t}"' for t in tokens)
        params: Dict[str, object] = {"q": match, "n": limit}
        scope = ""

        if doc_ids is not None:
            placeholders = ",".join(f":d{i}" for i in range(len(doc_ids)))
            scope = f" AND document_id IN ({placeholders})"
            params.update({f"d{i}": did for i, did in enumerate(doc_ids)})

        try:
            rows = self.db.execute(
                text(
                    f"SELECT id FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :q{scope} "
                    f"ORDER BY bm25({FTS_TABLE}) LIMIT :n"
                ),
                params,
            ).all()
        except Exception:
            return []

        return [r[0] for r in rows]

    @staticmethod
    def rrf_scored(
        ranked_lists: List[List[str]], top_k: int = 5, rrf_k: int = 60
    ) -> List[tuple]:
        """Reciprocal rank fusion over id rankings, with fused scores."""
        scores: Dict[str, float] = {}

        for ranking in ranked_lists:
            for rank, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        conversation_id: Optional[UUID] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Vector + BM25 candidates fused by RRF, hydrated from sqlite."""
        depth = candidate_k or self.candidate_k
        fusion_k = rrf_k if rrf_k is not None else self.rrf_k

        vector_ids = self.vector_search(query_vector, limit=depth, conversation_id=conversation_id)
        fts_ids = self.fts_search(query_text, limit=depth, doc_ids=doc_ids)

        if not vector_ids and not fts_ids:
            return []

        merged = self.rrf_scored([vector_ids, fts_ids], top_k=top_k, rrf_k=fusion_k)
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
                    "document_id": sid(row.document_id),
                    "index": row.index,
                    "heading": row.heading,
                    "page": row.page,
                    "text": row.text,
                    "score": score,
                }
            )

        return hits

    def _table_names(self) -> List[str]:
        """Table names; pinned lancedb returns a response with .tables."""
        return list(self.conn.list_tables().tables)

    def _table(self):
        """Open the chunks table, or None when it doesn't exist yet.

        Only a genuinely-absent table yields None; anything else raises
        so the caller sees the true failure instead of a bare None.
        """
        if "chunks" not in self._table_names():
            return None
        return self.conn.open_table("chunks")

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
