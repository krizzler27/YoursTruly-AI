# Memory System

This system remembers user facts across chats and older turns within a chat, so answers stay personal without refitting everything into the prompt. RAG means answering from uploaded files by retrieving relevant chunks first, and memory supplies the personal side of the same shared window. Rollup is the background job that summarizes older turns into episodic memory after a turn is saved, and compaction is the merge-and-forget pass that bounds episodic rows after rollup. The sections below cover the budget first, then the three tiers, the split, the lifecycle, and the tag plus project API.

Pointers only - detail lives elsewhere:

- docs/chat-pipeline.md - turn flow that calls into this budget
- docs/laya-integration.md - gates and scores consumed here
- docs/llama-inference-engine.md - summary engine used by rollup and compaction

## 1. Budget first - everything keys off it

One shared window serves history plus RAG plus memory. The window is
`EFFECTIVE_N_CTX`, resolved to 4096 below 12GB RAM and 8192 at or above
12GB, or an explicit `LLAMA_N_CTX` override
(`backend/config.py:71`). Nothing gets a fixed token constant outside
`allocate` (`backend/core/context_budget.py:52`).

`allocate(route, needs_memory, n_ctx, system_tokens, query_tokens)` order:

1. `total = window - SAFETY_MARGIN` (256 held back always, `backend/config.py:45`)
2. `answer` reserve by bin: 512 at 4K, 768 at 8K (`backend/config.py:46`)
3. `usable = total - answer - system_tokens - query_tokens`
4. memory carve: `needs_memory` selects 300 at 4K or 500 at 8K, else 0,
   clamped to `usable`; `rest = usable - mem` (`backend/config.py:52`)
5. route split on `rest`:
   - RAG: `rag_cap = rest * RAG share`, `history_cap = rest - rag_cap`
   - DIRECT: `history_cap = rest * HISTORY share`, `rag_cap = 0`

Bin table (`backend/config.py:45`, `backend/core/context_budget.py:52`):

- 4096: answer 512, mem 300, RAG share 0.7, HISTORY share 0.9, top_k 3
- 8192: answer 768, mem 500, RAG share 0.6, HISTORY share 0.8, top_k 5

Top_k is the number of retrieved chunks kept for the turn.

`top_k_for_ctx` returns 3 below 8192 and 5 at or above, unless an
explicit override is passed (`backend/core/context_budget.py:42`).
`count_tokens` uses the loaded chat engine tokenizer and falls back to
`ceil(len/4)` without force-loading (`backend/core/context_budget.py:14`).
`truncate_text` cuts char-proportionally then shrinks to fit
(`backend/core/context_budget.py:30`).

Worked example with defaults `system_tokens=300`, `query_tokens=50`:

- 4K DIRECT with memory: `total=3840`, `usable=2978`, `mem=300`,
  `rest=2678`, `history_cap=int(2678 * 0.9)=2410`, `rag_cap=0`
- 4K RAG with memory: same `rest=2678`,
  `rag_cap=int(2678 * 0.7)=1874`, `history_cap=804`
- 8K DIRECT with memory: `total=7936`, `usable=6818`, `mem=500`,
  `rest=6318`, `history_cap=int(6318 * 0.8)=5054`, `rag_cap=0`

Overflow-only policy (`backend/core/context_budget.py:108`,
`backend/core/context_budget.py:128`,
`backend/services/rag_service.py:37`,
`backend/services/rag_service.py:45`):

- `fit_history`: returns as-is when under cap; on overflow drops oldest
  first and always keeps the last 2 turns verbatim; reports truncated flag
- `fit_hits`: greedy score-ordered fill; a straddling hit is truncated to
  remaining room, then stop; reports truncated flag
- `_fit_memory`: truncates the memory block to `mem_cap`. The memory block is the combined semantic plus episodic text fitted into `mem_cap`.
- `_fit_topic`: sibling lines take only leftover `history_cap` after own
  history is fitted, never the memory carve

## 2. Three tiers - what lives where

Semantic - durable user facts shared across chats.
Schema: `SemanticMemoryModel` with unique indexed `key` and `value`
(`backend/db/models.py:82`). Write is `upsert` on a stripped lowercased
key, insert or replace (`backend/repository/semantic_repository.py:54`).
Read is `find_relevant` default limit 5; key-token overlap
(lowercase alnum tokens len 3 plus, `backend/repository/semantic_repository.py:14`)
ranks first, cosine vector hits at 0.5 or above
(`backend/repository/semantic_repository.py:11`) fill remaining result positions;
pool is `list_all` newest-first capped at 100
(`backend/repository/semantic_repository.py:69`); embed text clipped to
200 chars per candidate (`backend/repository/semantic_repository.py:19`).

Inspect semantic rows (columns verified against `backend/db/models.py:82`):

```sql
SELECT key, value, updated_at
FROM semantic_memory
ORDER BY updated_at DESC
LIMIT 5;
```

```sql
SELECT key, value
FROM semantic_memory
WHERE key LIKE '%prefer%';
```

Episodic - per-chat summaries of older turns.
Schema: `EpisodicMemoryModel` with `conversation_id` FK cascade,
`summary`, `turn_start`, `turn_end`, `recall_count` default 0,
nullable `last_recalled_at`; index on `(conversation_id, created_at)`
(`backend/db/models.py:95`). Write appends one row covering
`[start, start + len(head))` where `start` is the max stored `turn_end`
(`backend/services/episodic_service.py:40`,
`backend/repository/episodic_repository.py:17`). Read uses
`list_recent_for_query` newest-first pool of 10 as the recall candidate
set (`backend/repository/episodic_repository.py:68`); `recall` default
limit 3 ranks by token overlap with recency breaking ties, vector-only
hits filling past overlap hits (`backend/services/episodic_service.py:97`);
hits bump `recall_count` via `mark_recalled` in one commit
(`backend/repository/episodic_repository.py:80`). Row cap 20 triggers
compaction (`backend/services/episodic_service.py:26`).

Inspect episodic rows (columns verified against `backend/db/models.py:95`):

```sql
SELECT turn_start, turn_end, recall_count, last_recalled_at,
       substr(summary, 1, 80) AS summary_head
FROM episodic_memory
WHERE conversation_id = '<uuid>'
ORDER BY created_at DESC
LIMIT 10;
```

```sql
SELECT COUNT(*) AS rows,
       MIN(turn_start) AS first_turn,
       MAX(turn_end) AS last_turn
FROM episodic_memory
WHERE conversation_id = '<uuid>';
```

Manual tag - ad-hoc sharing key across sibling chats.
Tags are explicit, never inferred: a chat shares only when tagged
deliberately, and there is no auto-title or auto-populate
(owner decision - improvised labels fragment and pollute recall).
Schema: `ConversationsModel.tag`, nullable indexed `String(64)`.
Write is PATCH tag; blank or null clears;
normalized to strip plus 64-char cap. Read is `_tag_sibling_lines`:
same-tag siblings newest-first (limit 10), latest 1 episodic summary
each, max 3 labeled lines of form `Earlier in project {tag}: ...`;
untagged chats return no lines.
Caps: max 3 lines, fitted into
history-cap remainder via `_fit_topic`; own history is fitted first so
own-chat episodic in the memory carve keeps priority.

Projects - durable shared groups with one summary row.
A project is created in the UI and chats join by link
(`conversations.project_id`, nullable, opt-in). The background worker
aggregates member episodic rows into `projects.summary` after rollups.
Member chats read one labeled line (`Project {name} summary: ...`)
ahead of sibling lines inside the history remainder only.

Inspect tags:

```sql
SELECT id, title, tag
FROM conversations
ORDER BY updated_at DESC
LIMIT 10;
```

```sql
SELECT id, title
FROM conversations
WHERE tag = 'laya'
ORDER BY updated_at DESC
LIMIT 10;
```

## 3. Dynamic split - one mem_cap, two stores

Evidence triple `MemoryEvidence(needs_memory, sem_hit, epi_hit)` from
`memory_evidence` (`backend/services/decider.py:256`,
`backend/services/decider.py:308`): the decider is the component that picks DIRECT or RAG for the turn. Laya needs-memory at or above 0.8
(`backend/config.py:60`), OR stub hit (keywords, history prefs, semantic
key overlap), OR episodic token overlap over the recent 10 rows
(`backend/services/decider.py:253`). No vector is computed at gate time;
callers may pass pre-fetched `epi_rows` to skip the pool fetch
(`backend/services/rag_graph.py:451`).

`_memory_shares` splits `mem_cap` with a 0.7 dominant share
(`backend/services/rag_graph.py:30`,
`backend/services/rag_graph.py:162`): `mem_cap` is the token budget for the memory block, also called the memory carve.

- epi-only hit: episodic fits first up to 70 pct, semantic takes remainder
- sem-only hit: semantic fits first up to 70 pct, episodic takes remainder
- both or neither: semantic-first on the full cap

`_combine_memory` applies that split through `_fit_memory` and joins the
non-empty blocks (`backend/services/rag_graph.py:180`). Single-side input
takes the full cap. The limit is the passed cap, defaulting to
`EFFECTIVE_MEMORY_TOKENS` (`backend/config.py:101`). The same evidence
drives `allocate(needs_memory=...)` in `build_messages`
(`backend/services/rag_service.py:244`), so strong memory evidence keeps
its carve before RAG and history are fitted.

## 4. Lifecycle - rollup then compact

Rollup queue (`backend/services/rollup_job.py:26`,
`backend/services/rollup_job.py:38`):

- `chat_api` saves the assistant reply, then `RollupJob.submit` enqueues
  one `(conversation_id, request_id)` tuple and returns in microseconds
  with no model touch and no engine acquire
- One daemon worker takes jobs FIFO, opens its own `SessionLocal`,
  reloads history (limit 100), calls `rollup_if_needed`, then
  `compact_if_needed`
- `ChatServices.maybe_rollup` is the sync back-compat path
  (`backend/services/chat_services.py:90`); only the cheap no-trigger
  path (history 2 or fewer) is foreground-safe
- Trigger: turn count hits the every-10 cadence, or history tokens exceed
  `SUMMARY_TRIGGER` (0.85, `backend/config.py:54`) times usable tokens
  (`backend/services/episodic_service.py:87`); the head (all but the
  kept last 2) is summarized via `summarize_text` at max 128 tokens
  (`backend/services/episodic_service.py:26`)

Rollup trigger demo - turn counts with the same `usable` budget:

- 9 turns, tokens under 0.85 times usable: no trigger
  (`len <= 2` fast return misses, `_triggered` cadence misses,
  token check misses)
- 10 turns, even with tiny tokens: trigger
  (`len >= 10 and len % 10 == 0`, `backend/services/episodic_service.py:87`)
- 11 turns, tokens under threshold: no trigger
  (cadence misses, token check misses)
- 11 turns, tokens over 0.85 times usable: trigger
  (token branch of `_triggered` fires)
- 20 turns: trigger again on cadence; worker then runs
  `compact_if_needed` in the same pass (`backend/services/rollup_job.py:38`)

The worker reloads with `get_history(limit=100)` and derives `usable`
from `allocate(route="DIRECT", needs_memory=False, query_tokens=0)`
(`backend/services/rollup_job.py:38`), so the token branch keys off the
same math as section 1.

Compaction merge plus forget rules with protections
(`backend/services/episodic_service.py:160`):

- Runs after rollup in the worker; no-op at 20 rows or fewer
  (`backend/services/episodic_service.py:26`)
- Merge: exactly one oldest adjacent pair per call, never touching the
  newest 3 (`backend/services/episodic_service.py:26`); merged via
  worker-slot summary at max 128 tokens; a slot is one model holder with one role where only one generation runs at a time, and the worker slot is the transient WorkerEngine slot for background summaries. Aborts on empty output or token
  growth (`backend/services/episodic_service.py:215`); the new row
  carries min start and max end, copies the oldest parent `created_at`
  so created_at ordering stays chronological, and carries max
  `recall_count` and `last_recalled_at`; create-first then delete parents
  so failure keeps both
- Forget: only when still over cap; deletes zero-hit rows oldest-first
  outside the newest-10 pool (`backend/services/episodic_service.py:26`),
  up to the over-cap count (`backend/services/episodic_service.py:284`);
  the just-merged row id is excluded from the same-pass forget

## 5. Tag and project API

Tags are short deliberate labels (`laya`, `taxes-2026`) for ad-hoc
sharing across chats. Untagged (NULL) chats behave exactly as before the
feature existed. Tags are never generated from titles or first messages.

Contract: `ConversationUpdateRequest.tag` max 64 chars, null or blank
clears. `ChatServices.set_tag` calls `ChatRepository.set_tag` with the
same normalize rule.

```bash
PATCH /api/conversations/<uuid>
Content-Type: application/json

{"tag": "laya"}
```

- Set: `{"tag": "laya"}` tags the chat; value is stripped and capped
  at 64 chars
- Clear: `{"tag": null}` or `{"tag": "  "}` clears to NULL
- Title-only PATCH omits `tag`; the handler checks
  `model_fields_set` so the tag is left alone
- Sibling read: `list_by_tag(tag, limit=10, exclude_id=...)`
  newest-first minus the current chat
- Link: `{"project_id": "<uuid>"}` joins the chat to a project;
  `{"project_id": null}` clears the link

Delete cascade (`backend/services/chat_services.py:72`):
`delete_conversation` removes attached docs plus vectors and chunks, then
episodic rows via `delete_by_conversation`
(`backend/repository/episodic_repository.py:98`), then the conversation
row (messages follow the FK cascade, `backend/db/models.py:20`).

## 6. Reading the budget log

`build_messages` logs one line per turn in both branches
(`backend/services/rag_service.py:244`):

```text
budget total=3840 usable=2978 history=1200 rag=0 mem=300 mem_used=210 topic_used=40 route=DIRECT overflow=False
```

Field guide - each field maps to code:

- `total`: `allocate` total, `window - SAFETY_MARGIN`
  (`backend/core/context_budget.py:52`)
- `usable`: `allocate` usable, `total - answer - system - query`
- `history`: fitted history tokens, second value from `fit_history`
  (`backend/core/context_budget.py:108`)
- `rag`: fitted hit tokens, third value from `fit_hits`; always 0 on
  DIRECT (`backend/core/context_budget.py:128`)
- `mem`: `allocate` mem_cap, 0 when `needs_memory` is false
- `mem_used`: second value from `_fit_memory`
  (`backend/services/rag_service.py:37`)
- `topic_used`: second value from `_fit_topic`, bounded by
  `history_cap - hist_used` (`backend/services/rag_service.py:45`)
- `route`: `DIRECT` or `RAG` branch taken in `build_messages`
- `overflow`: `hist_truncated or hits_truncated`; True means oldest
  turns or lowest-score hits were dropped or cut

If `mem=0` but you expected memory, the gate returned
`needs_memory=False` - check `memory_evidence` inputs, not the fitter.
If `topic_used=0` with tagged siblings, own history already filled
`history_cap`, or siblings have no episodic rows yet.

## 7. Failure modes

Each failure fails open to a cheaper correct path. Fail-open means falling back to the safe-default instead of crashing the turn.

- Chat engine absent: `count_tokens` uses `ceil(len/4)`
- Graph, decider, or retrieval error: DIRECT with plain messages; RAG
  with zero hits reroutes to DIRECT
- Laya route, memory, hit-grade, or rewrite failure: SLM path, fused hit
  order, advisory grades, or original query text
- Memory pool or recall error: overlap-only recall, semantic-only text,
  or None; a memory-less turn never touches the embed engine; embed shape
  mismatch falls back to overlap-only
- Episodic busy engine, timeout, empty summary, or model error: rollup
  returns None and the caller truncates oldest instead
- Compaction merge abort or write error: both parents kept; forget or
  recall-touch errors log at debug and keep rows
- Rollup submit or worker error, unknown conversation: skipped; the next
  turn truncates oldest instead
- Tag sibling error or untagged chat: empty line list, prompt unchanged
- Migration inspect or column failure: skipped with warning, startup
  continues; fresh `create_all` plus migrate is a no-op

Upgrade note: `Base.metadata.create_all` creates missing tables but never
alters existing ones, so databases created before a column existed
failed on upgrade. Lifespan wiring runs
`create_all` first, then `ensure_schema(engine)`. `ensure_schema`
inspects each mapped table, issues `ALTER TABLE ADD COLUMN` for model
columns missing on disk, creates the index for indexed columns, is
idempotent, returns the added `table.column` list, and never raises.

## 8. Test map

- `backend/tests/test_episodic.py`: rollup trigger and head rules, recall
  ranking and touch counts, merge plus forget protections, rollup queue path
- `backend/tests/test_rag.py`: budget allocate and fit behavior, tag
  set and `list_by_tag`, message building with memory and topic blocks,
  sibling-line caps
- `backend/tests/test_migrate.py`: missing-column backfill, idempotent
  rerun, fresh-database no-op
- `backend/tests/test_laya.py`: route, memory, hit-grade, and rewrite gates
  plus thresholds consumed by the decider and graph
- `backend/tests/test_rag_eval.py` and `backend/tests/test_worker.py`:
  retrieval quality and summary engine behavior backing rollup and merge
