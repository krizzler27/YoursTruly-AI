# Chat Pipeline - Single Turn Debug Guide

This pipeline handles one user message from request to streamed reply, choosing plain or file-grounded answering and keeping the turn safe when anything fails. RAG means answering from uploaded files by retrieving relevant chunks first, and it is the file-grounded branch of this pipeline. Replies stream over SSE, which is server-sent events where the server pushes text chunks as they arrive. Use this doc to follow the per-turn order, check the streaming contract, and run copy-paste debug commands, with background work explained last.

Detail lives elsewhere by pointer only:
- gates (route, rewrite, grade, ground) - `docs/laya.md`
- budget shares, memory blocks, episodic rollup - `docs/memory-system.md`. Rollup is the background job that summarizes older turns into episodic memory after a turn is saved.
- chat and worker slots, streaming, model health - `docs/inference-engine.md`. A slot is one model holder with one role where only one generation runs at a time.
- retrieval, ingest, chunk layout - `docs/rag.md`

## 1. Per-turn flow - read this first

One `POST /api/chat/stream` runs this order. Refs are the claim.

1. Busy guard - reject when the chat slot is generating with HTTP 429 JSON - `backend/router/chat_api.py:28-33`.
2. Resolve conversation - `ensure_conversation` creates when id is None or reuses, retitles only when the stored title is blank or `New chat` - `backend/router/chat_api.py:35-41` and `backend/services/chat_services.py:25-41`.
3. Bind log context, load history, persist user turn - `set_conversation_id` binds `[conv=]` suffix, `get_history` loads last 5, `add_message` stores role `user` - `backend/router/chat_api.py:43-45`, `backend/core/logging.py:25-27`, `backend/services/chat_services.py:112-115`.
4. Optional model switch - `engine.switch_model` only when `request.model` is non-blank - `backend/router/chat_api.py:49-50`.
5. Dispatch - `start_time` starts TTFT here so it covers agent work, `run_agentic` runs in a thread - `backend/router/chat_api.py:52-57` and `backend/services/chat_services.py:127-135`.
6. Graph wiring - `decide` branches to `retrieve` on RAG else `build` - `backend/services/rag_graph.py:252-262`.
7. Decide node - decider call on query plus last 2 history turns, any exception falls back to DIRECT - `backend/services/rag_graph.py:303-315`. The decider is the component that picks DIRECT or RAG for the turn. Gate policy lives in `docs/laya.md`, not here.
8. Retrieve node - rewrite runs before any generation acquire, `rag.search` is scoped by conversation, error flips to DIRECT with reason `retrieval error`, zero hits flips to DIRECT with reason `no hits` - `backend/services/rag_graph.py:417-449` and `backend/services/rag_graph.py:347-415`. Retrieval detail lives in `docs/rag.md`.
9. Build node - memory text plus topic sibling lines feed one builder: RAG-with-hits builds grounded messages, otherwise plain chat - `backend/services/rag_graph.py:615-645` and `backend/services/rag_service.py:244-359`. Budget and memory detail lives in `docs/memory-system.md`.
10. Start generation - `llm.astream_chat` with temperature 0.5 on RAG else 0.6 - `backend/router/chat_api.py:59-64`. Slot behavior lives in `docs/inference-engine.md`.
11. Gate branch - `should_gate` is true only for RAG-with-hits - `backend/router/chat_api.py:66` and `backend/services/grounding_service.py:38-40`. Grade policy lives in `docs/laya.md`.
12. Direct path - pull first delta for TTFT, wrap as SSE, persist assistant reply, submit rollup, emit `[DONE]` - `backend/router/chat_api.py:78-141`.
13. Gated path - `_gated_response` resolves filenames, buffers the draft, grades, streams buffered parts in order on pass or skip and refusal on fail, persists the sent text, submits rollup, emits `[DONE]` - `backend/router/chat_api.py:149-222` and `backend/services/grounding_service.py:143-168`.
14. CRUD outside turns - create, list, list messages, rename plus topic, delete with cascade - `backend/router/chat_api.py:225-287`. Delete removes docs and episodic rows in code and messages via FK cascade - `backend/services/chat_services.py:72-88` and `backend/db/models.py:41`.
15. Schemas - `ChatRequest` is `query` plus optional `model`, `conversation_id`, and `context`; empty query is rejected by validation - `backend/schemas/api_schemas.py:6-17`. Note: the router never reads `request.context`, so it is accepted but has no effect on the turn.

## 2. SSE contract

- Success is `StreamingResponse` with `media_type` text/event-stream - `backend/router/chat_api.py:131-141` and `backend/router/chat_api.py:212-222`.
- Headers on both direct and gated streams:
  - `X-TTFT` - ms from dispatch start to first visible byte, gated includes grade time - `backend/router/chat_api.py:135` and `backend/router/chat_api.py:216`.
  - `X-Route` - final `DIRECT` or `RAG` - `backend/router/chat_api.py:137` and `backend/router/chat_api.py:218`.
  - `X-Conversation-Id` - resolved conversation id - `backend/router/chat_api.py:136` and `backend/router/chat_api.py:217`.
  - `Cache-Control: no-cache` and `X-Accel-Buffering: no` - `backend/router/chat_api.py:138-139`.
  - CORS exposes `X-Conversation-Id`, `X-TTFT`, `X-Route`, `X-Request-Id` - `backend/main.py:65`.
- Body frames:
  - `data: {"content": "<delta>"}` one or more times - `backend/router/chat_api.py:105-111` and `backend/router/chat_api.py:195-196`.
  - `data: [DONE]` is always last on a 200 stream - `backend/router/chat_api.py:129` and `backend/router/chat_api.py:210`.
- Error shapes:
  - Pre-stream busy - HTTP 429 JSON `{"detail": "System Busy - model is generating. Try again."}` - `backend/router/chat_api.py:30-33`.
  - Mid-setup failure - HTTP 429 or 500 JSON with `X-Conversation-Id`, 500 shape is `{"error", "type", "request_id", "hint"}` - `backend/router/chat_api.py:87-97`.
  - Mid-stream failure direct - SSE `data: {"error": "<msg>"}` then `data: [DONE]`, reply may be partial and unsaved - `backend/router/chat_api.py:112-114`.
  - Gated setup failure - SSE `data: {"error": "<msg>"}` then `data: [DONE]` with TTFT headers - `backend/router/chat_api.py:174-188`.
  - Save failure after full display - SSE `data: {"warning": "not saved"}` before `[DONE]` - `backend/router/chat_api.py:120-122` and `backend/router/chat_api.py:201-203`.
  - Unknown conversation - HTTP 404 JSON `{"detail": "Conversation <id> not found"}` - `backend/router/chat_api.py:40-41` and `backend/services/chat_services.py:34-36`.

## 3. Debug runbook - copy-paste commands

Run the server first, then run these against `http://127.0.0.1:8000`. Flag shapes match the router source. These were verified against source, not executed.

### 3.1 Health and one streaming turn

Health:

```sh
curl -i http://127.0.0.1:8000/api/health
```

- Source: `GET /api/health` - `backend/main.py:72-87`.
- Healthy means `status` is `ready`; otherwise the message points to Explore download.

New chat turn with header capture:

```sh
curl -N -D - -o chat.sse http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d "{\"query\": \"what does the attached doc say about refunds?\"}"
```

Follow-up in the same conversation:

```sh
curl -N -D - -o chat2.sse http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d "{\"query\": \"what about the second page?\", \"conversation_id\": \"<CONV_ID>\"}"
```

- Source: `POST /api/chat/stream` with `ChatRequest` - `backend/router/chat_api.py:24-25` and `backend/schemas/api_schemas.py:6-17`.
- `-N` disables curl buffering so SSE deltas show as they arrive.
- `-D -` prints response headers to stdout so you can read `X-TTFT`, `X-Route`, and `X-Conversation-Id` while `-o` saves the body frames.
- Empty `query` is rejected by validation, so always send non-blank text - `backend/schemas/api_schemas.py:12-17`.
- Optional `model` switches the chat model for this turn - `backend/router/chat_api.py:49-50`.

### 3.2 Conversation CRUD and doc upload

Create a shell before attaching files:

```sh
curl -i http://127.0.0.1:8000/api/conversations -H "Content-Type: application/json" -d "{\"title\": \"debug chat\"}"
```

List, read, rename plus topic, delete:

```sh
curl -i http://127.0.0.1:8000/api/conversations
curl -i http://127.0.0.1:8000/api/conversations/<CONV_ID>/messages
curl -i -X PATCH http://127.0.0.1:8000/api/conversations/<CONV_ID> -H "Content-Type: application/json" -d "{\"title\": \"refund debug\"}"
curl -i -X PATCH http://127.0.0.1:8000/api/conversations/<CONV_ID> -H "Content-Type: application/json" -d "{\"topic\": \"refunds\"}"
curl -i -X DELETE http://127.0.0.1:8000/api/conversations/<CONV_ID>
```

- Sources: create, list, list messages, patch, delete - `backend/router/chat_api.py:225-287`; title and topic shapes - `backend/schemas/api_schemas.py:59-78`.
- `topic` accepts a short tag, null or blank clears it.

Upload one doc to a conversation, then list docs:

```sh
curl -i http://127.0.0.1:8000/api/ingest -F "file=@./notes.md" -F "conversation_id=<CONV_ID>"
curl -i "http://127.0.0.1:8000/api/documents?conversation_id=<CONV_ID>"
```

- Source: `POST /api/ingest` takes `file` plus `conversation_id` form fields and returns 202 - `backend/router/rag_api.py:43-97`.
- Accepted suffixes are `.txt`, `.md`, `.pdf` - `backend/services/ingest_service.py:24`.
- Source: `GET /api/documents` with optional `conversation_id` - `backend/router/rag_api.py:100-113`.

### 3.3 SQLite reads

The app uses SQLite via `sqlite+pysqlite:///yourstrulyai.db` - `backend/config.py:43` and `backend/db/db_engine.py:8-13`. From the repo root the file is `backend/yourstrulyai.db`; from `backend/` it is `yourstrulyai.db`. Table names are `messages`, `documents`, `document_chunks` - `backend/db/models.py:35-80`.

Last turns for one conversation:

```sh
sqlite3 backend/yourstrulyai.db "SELECT role, substr(content, 1, 80), created_at FROM messages WHERE conversation_id = '<CONV_ID>' ORDER BY created_at DESC LIMIT 10;"
```

Doc rows and index state for one conversation:

```sh
sqlite3 backend/yourstrulyai.db "SELECT filename, status, chunk_count FROM documents WHERE conversation_id = '<CONV_ID>' ORDER BY created_at DESC LIMIT 10;"
```

First chunks for one doc:

```sh
sqlite3 backend/yourstrulyai.db "SELECT \"index\", heading, page, substr(text, 1, 120) FROM document_chunks WHERE document_id = '<DOC_ID>' ORDER BY \"index\" LIMIT 5;"
```

- Quote `"index"` because it is a keyword.
- If a RAG turn has no hits, check here first: either no `documents` row is indexed for the conversation or no `document_chunks` rows match.

### 3.4 Log grep keys - which line proves each stage

Log format is `timestamp - [name: level] - message` plus `[req=]` and `[conv=]` suffixes when bound - `backend/core/logging.py:55-77`. Conversation id is bound at turn start - `backend/router/chat_api.py:43`.

| Stage | Grep for | What it proves | Source |
| --- | --- | --- | --- |
| dispatch | `Chat dispatch started` and `Chat ready - route=` | query entered the graph, final route and hit count | `backend/services/chat_services.py:132-134` |
| decide | `RAG decision - route=` | final route, hit count, reason | `backend/services/rag_graph.py:295-300` |
| rewrite | `rewrite=` and `rewrite_skipped` | `true` with token delta means the query was expanded, `false` or `skipped` means search used the original | `backend/services/rag_graph.py:365-415` |
| budget | `budget total=` | usable, history, rag, mem caps and overflow for DIRECT vs RAG | `backend/services/rag_service.py:274-285` and `backend/services/rag_service.py:305-316` |
| memory | `memory split` | sem vs epi carve and tokens used | `backend/services/rag_graph.py:212-219` |
| TTFT | `TTFT` | ms from dispatch start to first visible byte, gated includes grade time | `backend/router/chat_api.py:81` and `backend/router/chat_api.py:191` |
| grounding | `grounding gate grounded=` | `True` streams draft, `False` substitutes refusal, `None` skips fail-open | `backend/router/chat_api.py:192` and `backend/services/grounding_service.py:95-124` |
| rollup | `rollup` and `episodic compact done` | submit accepted the job, worker compacted or skipped fail-open | `backend/services/rollup_job.py:32-66` |
| fail-open | `Graph failed`, `Decider failed`, `Retrieval failed` | graph, decider, or search crashed and fell back to DIRECT | `backend/services/rag_graph.py:285`, `backend/services/rag_graph.py:312`, `backend/services/rag_graph.py:436` |

Compaction is the merge-and-forget pass that bounds episodic rows after rollup.

## 4. Failure modes - what the user sees

- 429 busy - chat slot already generating at entry, during first delta, or in gated setup.
  - User-visible result: HTTP 429 JSON or SSE error frame, no partial assistant save - `backend/router/chat_api.py:28-33` and `backend/router/chat_api.py:87-92` and `backend/router/chat_api.py:163-169`.
- Retrieval error - `rag.search` throws inside the retrieve node.
  - User-visible result: route flips to DIRECT with reason `retrieval error`, plain chat answer with no citations - `backend/services/rag_graph.py:435-440`.
- Empty hits - RAG decision returns zero hits.
  - User-visible result: route flips to DIRECT with reason `no hits`, plain answer with no citations - `backend/services/rag_graph.py:444-447`. The grounded prompt is not used here; the build path is plain chat - `backend/services/rag_graph.py:635-644`.
- Gate errors - Laya down, malformed grounding shape, filename lookup miss, or empty grade input.
  - User-visible result: gate skips and streams the draft unchanged, filenames fall back to raw ids or empty map, no refusal shown - `backend/services/grounding_service.py:43-65` and `backend/services/grounding_service.py:95-124`.
- Graph or decider crash - any unhandled error in `RagGraph.run` or the decide node.
  - User-visible result: fail-open DIRECT with plain history plus query - `backend/services/rag_graph.py:284-291` and `backend/services/rag_graph.py:303-315`.

## 5. Background work - after the turn

- Rollup - both stream paths call `rollup_job.submit` after a saved assistant reply; the worker reloads history and summarizes off the request path, everything fail-open - `backend/router/chat_api.py:124-127`, `backend/router/chat_api.py:204-208`, `backend/services/rollup_job.py:26-68`. Budget for rollup lives in `docs/memory-system.md`.
- Ingest - upload returns 202 fast and indexes in a background FIFO; a turn right after upload can still be DIRECT until chunks land - `backend/router/rag_api.py:43-97`. Chunk layout lives in `docs/rag.md`.
