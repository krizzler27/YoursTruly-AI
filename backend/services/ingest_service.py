"""Ingest: file → chunk → embed → hybrid store. Only ingest() is public."""

from pathlib import Path
from typing import List, Optional

import uuid
import uuid6
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sqlalchemy.orm import Session

from config import config
from db.models import DocumentChunksModel, DocumentsModel
from repository.lance_repository import LanceRepository
from schemas.rag_schemas import Chunk
from services.llama_engine import EmbeddingEngine

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
MAX_FILE_MB = 25


def stored_upload_path(conversation_id: uuid.UUID, filename: str) -> Path:
    """App-owned copy location: <models-dir>/../docs/<chat>/<file>."""
    safe = Path(filename).name.strip()
    target = Path(config.LLAMA_MODEL_PATH).parent / "docs" / str(conversation_id)
    target.mkdir(parents=True, exist_ok=True)
    return target / safe

MARKDOWN_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class IngestService:
    def __init__(
        self,
        db: Session,
        engine: Optional[EmbeddingEngine] = None,
        lance: Optional[LanceRepository] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.db = db
        self.engine = engine or EmbeddingEngine.get_instance()
        self.lance = lance or LanceRepository(db)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---- load (one Document per txt/md file, one per pdf page) ----

    def load_source(self, source_path: str) -> List[Document]:
        """Read txt/md/pdf into langchain Documents."""
        path = Path(source_path)
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported file type: {suffix or path.name}")

        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(f"file exceeds {MAX_FILE_MB}MB: {path.name}")

        if suffix == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(str(path))
            return [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(path), "page": i},
                )
                for i, page in enumerate(reader.pages)
            ]
        return [
            Document(
                page_content=path.read_text(encoding="utf-8", errors="replace"),
                metadata={"source": str(path)},
            )
        ]

    # ---- chunk (one splitter per file type — md/txt/pdf differ) ----

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,
            chunk_overlap=self.chunk_overlap * 4,
            length_function=len,
            separators=SPLIT_SEPARATORS,
        )

    def chunk_md(self, text: str) -> List[Chunk]:
        """Markdown: split by headers, re-split long sections."""
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS, strip_headers=True
        )
        recursive = self._splitter()
        chunks: List[Chunk] = []
        for section in header_splitter.split_text(text):
            heading = " ".join(f"{k} {v}" for k, v in section.metadata.items())
            for piece in recursive.split_text(section.page_content):
                body = piece.strip()
                if not body:
                    continue
                chunks.append(
                    Chunk(
                        text=f"{heading}\n\n{body}" if heading else body,
                        heading=heading,
                        index=len(chunks),
                    )
                )
        return chunks

    def chunk_txt(self, text: str) -> List[Chunk]:
        """Plain text: recursive split only, no headings."""
        chunks: List[Chunk] = []
        for piece in self._splitter().split_text(text):
            if piece.strip():
                chunks.append(Chunk(text=piece.strip(), index=len(chunks)))
        return chunks

    def chunk_pdf(self, docs: List[Document]) -> List[Chunk]:
        """PDF pages: recursive split per page, page number preserved."""
        chunks: List[Chunk] = []
        for doc in docs:
            for piece in self._splitter().split_text(doc.page_content):
                if piece.strip():
                    chunks.append(
                        Chunk(
                            text=piece.strip(),
                            index=len(chunks),
                            page=doc.metadata.get("page"),
                        )
                    )
        return chunks

    def chunk_and_embed(self, docs: List[Document], suffix: str) -> List[Chunk]:
        """Split by file type, then embed — the ingest path's single call."""
        if suffix == ".md":
            chunks = self.chunk_md(docs[0].page_content)
        elif suffix == ".pdf":
            chunks = self.chunk_pdf(docs)
        else:
            chunks = self.chunk_txt(docs[0].page_content)
        vectors = self.engine.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"embed returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        return chunks

    # ---- ingest ----

    def ingest(
        self, source_path: str, filename: str, conversation_id=None
    ) -> DocumentsModel:
        """Index one txt/md/pdf file into one chat. Re-ingesting a filename replaces it.

        Insert-new-first under a temp filename, single commit for all
        sqlite writes; old vectors drop after the commit (LanceDB can't
        join it). Any earlier failure rolls back and sweeps new vectors.
        """
        suffix = Path(source_path).suffix.lower()
        docs = self.load_source(source_path)
        chunked = self.chunk_and_embed(docs, suffix)
        if not chunked:
            raise ValueError(f"no chunks produced from {filename}")

        existing = (
            self.db.query(DocumentsModel)
            .filter(
                DocumentsModel.filename == filename,
                DocumentsModel.conversation_id == conversation_id,
            )
            .first()
        )
        old_id = existing.id if existing is not None else None

        doc = DocumentsModel(
            filename=f"__pending__{uuid6.uuid7().hex}",
            chunk_count=len(chunked),
            status="indexed",
            conversation_id=conversation_id,
        )
        self.db.add(doc)
        self.db.flush()

        try:
            self.lance.upsert_chunks(doc.id, chunked, commit=False, conversation_id=conversation_id)
            self.db.add_all(
                [
                    DocumentChunksModel(
                        id=f"{doc.id}:{c.index}",
                        document_id=doc.id,
                        index=c.index,
                        heading=c.heading,
                        page=c.page,
                        text=c.text,
                    )
                    for c in chunked
                ]
            )
            if old_id is not None:
                self.lance.delete_fts(old_id)
                self.db.delete(existing)
                self.db.flush()  # DELETE before rename (unique filename)
            doc.filename = filename
            self.db.commit()
        except Exception:
            self.db.rollback()
            try:
                self.lance.delete_vectors(doc.id)
            except Exception:
                pass
            raise

        if old_id is not None:
            self.lance.delete_vectors(old_id)
        return doc
