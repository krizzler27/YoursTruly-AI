# RAG System: Hybrid Retrieval over Per-Chat Uploads

Chat needs file-grounded answers from per-chat uploads (txt/md/pdf) without evicting the resident chat model. RAG means answering from uploaded files by retrieving relevant chunks first, and here it embeds the query once, fuses vector plus keyword candidates, grades with Laya, and fits hits into the shared window. Ingest chunks, embeds, and stores the same files ahead of time. Use this doc to operate retrieval and ingest, with tuning knobs and failure modes below.

- Turn flow (decide, retrieve, build, gated stream): docs/chat-pipeline.md
- Grading, rewrite, and grounding gates plus thresholds: docs/laya.md sections 1.4 and 6.2
- Shared budget math, memory tiers, topic lines: docs/memory-system.md

## Retrieval path (read this first)

`RagService.search()` then `build_messages()` fits hits into the shared window.

1. Embed once - backend/services/rag_service.py:105-138. `engine.embed([clean])` runs first. Empty query raises `ValueError`; empty vector raises `RuntimeError`. Scoped chats resolve `doc_ids` after the single embed - no docs returns `[]` before any hybrid work.
2. Hybrid candidates - backend/repository/lance_repository.py:238-257. Depth `candidate_k = 20` per arm (backend/repository/lance_repository.py:64-75). Vector arm is L2 over unit-norm vectors with a `conversation_id` prefilter, so it ranks as cosine (backend/repository/lance_repository.py:157-180). FTS arm is BM25 over an OR of quoted word tokens scoped to `doc_ids`; reserved words are quoted so FTS5 syntax never breaks, and a missing table or bad match returns `[]` (backend/repository/lance_repository.py:182-223). Either arm may be empty; both empty returns `[]`.
3. RRF fusion - backend/repository/lance_repository.py:225-236. RRF is reciprocal rank fusion, a method that merges two ranked lists by rank position so a chunk scoring well in either arm rises. `sum(1 / (rrf_k + rank + 1))` per id across the two rankings with `rrf_k = 60`. Top `top_k` ids are hydrated from sqlite `DocumentChunksModel`, so orphan vectors from a hard kill drop silently and corrupt rows are skipped with a warning (backend/repository/lance_repository.py:262-296). Top_k is the number of retrieved chunks kept for the turn.
4. Laya grading tier - backend/services/rag_service.py:140-191. Exact(2) first, then partial(1), then irrelevant(0), stable by fused score within a level. Level-0 drops only when a level 1 or above exists, so grading alone never empties the list. Below `LAYA_HIT_THRESHOLD` the head is advisory and fused order is kept. Any shape, count, or inference error raises inside `_grade_hits` (backend/services/rag_service.py:67-86) and `search()` keeps fused order. Model, threshold, and template detail: docs/laya.md sections 1.4 and 6.2.
5. Top_k bins - backend/core/context_budget.py:42-49. Requested `top_k` (POST body default 5, range 1-20 per backend/schemas/rag_schemas.py:28-33) bounds fusion depth; `_grade_hits` truncates the graded list to the bin below.

| EFFECTIVE_N_CTX | top_k | Source bin |
| --- | --- | --- |
| 4096 | 3 | below 8192 |
| 8192 and above | 5 | at or above 8192 |

An explicit override wins when passed. Window resolution (override else 4096 below 12 GB RAM, 8192 at or above): backend/config.py:71-76.

6. Budget fit - backend/services/rag_service.py:244-359. `allocate(route="RAG")` splits one window into answer reserve, memory carve, rag cap, and history cap; `fit_hits` fills score-ordered and truncates the straddler; empty hits takes the DIRECT branch with no rag cap. Allocator math and share table: docs/memory-system.md. Citation labels (`[name:heading]` / `[name:pN]`): backend/services/rag_service.py:320-332.

## Ingest path (write path internals)

Flow: `POST /api/ingest` (multipart file plus `conversation_id`) returns 202 with a `pending` row, then a background FIFO worker runs `ingest_job` -> `IngestService.ingest()` (backend/router/rag_api.py:43-97, backend/services/ingest_job.py:69-124). Only `ingest()` is public on the service (backend/services/ingest_service.py:168-234). The per-file summary is written by the job after `ingest()` returns (backend/services/ingest_job.py:110-115), not inside `ingest()`.

- Load - backend/services/ingest_service.py:64-91. One `Document` per txt/md file, one per pdf page via pypdf. Guards: suffix in `{.txt, .md, .pdf}` (backend/services/ingest_service.py:24), 25 MB cap (backend/services/ingest_service.py:25), else `ValueError`. Staged path is `__incoming__<uuid>_<file>` under `<models-dir>/../docs/<chat>/` (backend/services/ingest_service.py:36-41); the canonical copy replaces it only on success (backend/services/ingest_job.py:110-113).
- Chunk - backend/services/ingest_service.py:95-146. Per-type splitters over `RecursiveCharacterTextSplitter` with `chunk_size * 4` / `chunk_overlap * 4` chars (defaults 512/50 give 2048/200). Md splits on H1-H4 headers with the heading prepended as breadcrumb then recursive re-split (backend/services/ingest_service.py:103-123); txt is recursive only with no headings (backend/services/ingest_service.py:125-131); pdf is recursive per page with page number preserved (backend/services/ingest_service.py:133-146).
- Embed - backend/services/llama_engine.py:191-236. Transient singleton, `embedding=True`, ctx clipped to `min(EFFECTIVE_N_CTX, 2048)`, `n_batch = n_ubatch = 2048`, `batch_size = 16`, `normalize=True`. A slot is one model holder with one role where only one generation runs at a time. Single call in `chunk_and_embed` with a length-mismatch guard (backend/services/ingest_service.py:148-164). Model resolves via `EFFECTIVE_EMBED_MODEL` (backend/config.py:116-124). Loads beside the chat slot briefly, unloads right after.
- Store - backend/repository/lance_repository.py:81-132. Two stores, one contract: LanceDB `backend/lancedb/` rows keyed `{doc_id}:{index}` (vector) and FTS5 `document_chunks_fts(id, document_id, text)` in the same sqlite file. Both write and delete together.
- Atomicity - backend/services/ingest_service.py:184-234. Insert-new-first under `__pending__<uuid>` filename, single commit for all sqlite writes (new doc row, chunk rows, FTS inserts with `commit=False`); old FTS plus row deleted pre-commit with a flush before rename; old vectors dropped post-commit. Earlier failure rolls back sqlite and sweeps new vectors via `delete_vectors` (backend/repository/lance_repository.py:148-154). LanceDB cannot join the sqlite transaction, so a crash between commit and post-commit drop leaves stale vectors (see Failure modes).
- Endpoints - backend/router/rag_api.py:23-137. `POST /api/ingest` (202), `POST /api/search` (sync in threadpool, unloads embed engine in `finally`), `GET /api/documents`, `DELETE /api/documents/{id}` (vectors plus FTS plus chunks plus stored copy via backend/services/rag_service.py:201-225).

## Operate it (copy-paste, then tune)

Ingest one file for one chat (202 means queued, not indexed):

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@notes.md" \
  -F "conversation_id=<CHAT_UUID>"
```

Shape per backend/router/rag_api.py:43-97: multipart `file` plus form `conversation_id`; bad suffix or missing name is 400; unknown chat is 404 via the job submit.

Search the same chat (body per backend/schemas/rag_schemas.py:28-33):

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what decided the vector store?", "top_k": 5, "conversation_id": "<CHAT_UUID>"}'
```

Empty query is 400; embed failure is 500 with `request_id` (backend/router/rag_api.py:23-40).

Sqlite checks against the app db (tables per backend/db/models.py:45-74 and backend/repository/lance_repository.py:26). Ids below are dashed UUIDs; chunk PKs are `<doc-id>:<index>` (backend/services/ingest_service.py:205-217):

```bash
sqlite3 backend/yourstrulyai.db \
  "SELECT filename, status, chunk_count FROM documents WHERE conversation_id = '<CHAT_UUID>';"

sqlite3 backend/yourstrulyai.db \
  "SELECT COUNT(*) FROM document_chunks WHERE id LIKE '<DOC_UUID>:%';"

sqlite3 backend/yourstrulyai.db \
  "SELECT COUNT(*) FROM document_chunks_fts WHERE document_id = '<DOC_UUID>';"
```

Vectors live outside sqlite in `backend/lancedb/` table `chunks` (backend/repository/lance_repository.py:64-75); a `pending` row with no chunk rows means the job has not indexed yet, `failed` names the reason (backend/services/ingest_job.py:117-124).

Recall eval reruns (no live server; stop it first so only the embed model is resident):

```bash
cd backend
uv run python rag_eval/retrieval-eval/gen_synthetic_corpus.py
uv run python rag_eval/retrieval-eval/eval_recall.py
```

Corpus is 30 markdowns with H1-H4 plus `qa.jsonl` with 40 questions (backend/rag_eval/retrieval-eval/eval_recall.py:28-32). The full live-model round (ingest plus graph plus generation, about 15-25 min on CPU) is backend/tests/test_rag_eval.py:1-21:

```bash
cd backend
.venv\Scripts\python.exe -m unittest tests.test_rag_eval -v
```

Real-eval artifacts (8 docs, 73 `qa.jsonl` rows, notebook harness): `backend/rag_eval/docs/`, `backend/rag_eval/qa.jsonl`, `backend/rag_eval/eval_real.ipynb`.

## Known baselines

Cite from `.devplans/rag.md` (dev-only tracker, not a user doc) sections 4 (synthetic) and 7 (real-eval round 1, 2026-09-10). Do not re-derive.

- Synthetic (30 markdowns with H1-H4, 40 questions, real nomic embed, temp stores): recall k=1 vector 0.88 / BM25 0.78 / hybrid 0.88; k=3 0.97 / 0.95 / 1.00; k=5 1.00 / 0.97 / 1.00. Hybrid matches or beats both arms at every k, zero misses at top 5. Phase 3 re-run identical.
- Real-eval round 1 shape: corpus 8 docs (3 PDF plus 3 MD plus 2 messy txt) in `backend/rag_eval/docs/`; `qa.jsonl` 73 rows (60 PDF/MD factual plus 11 txt plus 2 DIRECT chit-chat); harness `backend/rag_eval/eval_real.ipynb`. Hybrid recall @1 0.93, @3 0.99, @5 0.99 with 1 miss (#9, meta-text ranking gap). Ingest 68 s total (110-chunk PDF dominates). Timings: embed 47 ms/q, search 17 ms, build 4 ms, live 14.5 s p50 on 3B Q8 CPU. Gate 38/73 = 0.52 (prompt named zero files - fixed structurally by per-chat inventory, not by tuning). Grounded 18/71 in-context, 70/71 cited. Live 3/71 exact substring (strict metric plus paraphrase plus refusals; human read about 53 percent clean). Known label noise: #26 gold answer in neither doc, #50 answer in both docs.

## Failure modes

- Stale-vector window: crash between sqlite commit and post-commit old-vector drop leaves both old and new vectors visible; next re-ingest heals (delete-then-insert). Accepted V1 risk.
- Dim mismatch: embed model swapped mid-corpus (vector dim not equal to table dim, or mixed dims in one upsert) raises `ValueError` fail-fast in backend/repository/lance_repository.py:312-331. Fix is wipe `backend/lancedb/` and re-upload.
- Empty hits (all return `[]`, never error): empty or blank query raises `ValueError` (400 at the router); scoped chat with zero docs short-circuits; both arms empty returns `[]`; grading never empties (level-0 kept when nothing scores 1 or above; advisory path keeps fused order).
- Embed failure: `search()` raises (`RuntimeError` on no vector); router maps to 500 with request_id. Grading failure degrades to fused order (logged at debug). Restart-mid-ingest orphans the stored copy plus a non-live row - delete plus retry. Worker waits up to 60 s for an idle engine (`IDLE_TIMEOUT_S`, backend/services/ingest_job.py:33), then marks the row failed with a retry message (backend/services/ingest_job.py:77-81). There is no 409 on this path; ingest never blocks the request on the engine.

## Test map

All in `backend/tests/test_rag.py` unless noted; stubbed engines and temp DBs/Lance dirs, real `yourstrulyai.db` never touched.

- Store and fusion: `test_sid_chunk_id_roundtrip`, `test_doc_ids_match_fts_scope`, `test_reserved_words_do_not_raise`, `test_empty_and_scoped_empty`, `test_deterministic_merge`, `test_hybrid_fuses_vector_and_bm25`, `test_vector_respects_conversation_scope`.
- Chunkers and ingest atomicity: `test_empty_sources_yield_no_chunks`, `test_txt_and_md_split`, `test_reject_bad_suffix_and_size`, `test_ingest_replaces_same_name`.
- Budget and builder: `test_derived_not_fixed`, `test_build_messages_caps_and_cites_page` (rag cap plus `[name:heading]` / `[name:pN]` labels).
- RAG turn behavior (wiring detail in docs/chat-pipeline.md): `test_rag_route_searches_scoped`, `test_direct_skips_retrieval`, `test_retrieval_error_fails_open_direct`.
- Background jobs: `test_submit_*`, `test_failed_*`, `test_process_removes_temp_row_on_success`, `test_superseded_job_is_noop`.
- Eval harness: `backend/tests/test_rag_eval.py::test_1_recall_and_grounding`, `test_2_generation` (recall plus grounding shape, not the live-model gate).
