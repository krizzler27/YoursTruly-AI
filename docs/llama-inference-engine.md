# Inference engine - one engine, every model kind

This engine loads and unloads local models so chat, background summaries, and embeddings share one small box without running out of memory. A slot is one model holder with one role where only one generation runs at a time, and this doc owns slots, lifecycle, load settings, resolution, memory bins, and health. Use the runbooks below to add a model slot or change RAM bins. Consumer behavior lives elsewhere by pointer only: docs/chat-pipeline.md, docs/rag.md, docs/memory-system.md.

## 1. Slot table and lifecycle first

One registry holds one slot per role. Creation never loads weights. Only one generation runs per slot at a time.

| Slot | Class | Lifecycle | Who loads | Who unloads |
|------|-------|-----------|-----------|-------------|
| chat | `LlamaEngine` | resident | lifespan preload in `backend/main.py:32-37`, else `ensure_loaded` on first call | shutdown only in `backend/main.py:41-42` |
| worker | `WorkerEngine` | transient | `ensure_loaded` on demand in `backend/services/llama_engine.py:327-338` | caller in `finally` in `backend/services/summarize_service.py:117-128` and `backend/services/rag_graph.py:77-86` |
| embed | `EmbeddingEngine` | transient | `embed()` self-loads when empty in `backend/services/llama_engine.py:226-227` | owning `RagGraph.run` in `finally` in `backend/services/rag_graph.py:272-283` |

Lifecycle for every slot:

1. Create empty. `LlamaEngine.get_instance(role)` builds the slot under the shared lock and returns it without loading - `backend/services/llama_engine.py:52-60`. `__init__` sets `llm = None` - `backend/services/llama_engine.py:24-29`.
2. Load once. `ensure_loaded` calls `load` only when `is_loaded` is false and normalizes `FileNotFoundError` to `RuntimeError` - `backend/services/llama_engine.py:85-91`. `load` returns early when already loaded - `backend/services/llama_engine.py:93-96`, `backend/services/llama_engine.py:191-194`, `backend/services/llama_engine.py:327-330`.
3. Guard each generation. `acquire` does atomic check-and-set under `_gen_lock` and raises `RuntimeError("System Busy...")` when held - `backend/services/llama_engine.py:73-79`. `release` clears the flag - `backend/services/llama_engine.py:81-83`. `LLMService.invoke` uses `ensure_loaded`, `acquire`, try, finally `release` - `backend/services/llm_service.py:157-191`. `LLMService.astream_chat` uses the same shape with a worker thread for streaming - `backend/services/llm_service.py:91-92`, `backend/services/llm_service.py:98-139`.
4. Unload by owner only. `unload` deletes the handle, nulls it, runs `gc.collect()`, and logs - `backend/services/llama_engine.py:141-150`. Chat never unloads mid-process. Worker and embed unload after each use so only one weight set stays resident.
5. Never nest. Call `summarize_text` and query rewrite before acquiring the chat engine, never inside an acquire - `backend/services/summarize_service.py:1-13`, `backend/services/rag_graph.py:418-420`. The helpers take their own short chat-slot acquire or own the worker slot outright, so nesting re-enters the same guard and deadlocks - `backend/services/summarize_service.py:104-114`, `backend/services/rag_graph.py:66-86`.

## 2. Load anatomy

Chat and worker share generative load kwargs. Embed uses a clipped encoder-only load.

- Path pick. Slot path wins when set and present, else the config finder pick: `EFFECTIVE_CHAT_MODEL` for chat, `EFFECTIVE_EMBED_MODEL` for embed, `resolve_worker_model_path` for worker - `backend/services/llama_engine.py:93-101`, `backend/services/llama_engine.py:191-198`, `backend/services/llama_engine.py:327-338`.
- Generative kwargs. `n_ctx` from `EFFECTIVE_N_CTX`, threads and batch threads from `EFFECTIVE_N_THREADS`, `n_batch=512`, `n_ubatch=256`, Q4_0 KV cache types, `flash_attn=True`, `use_mmap=True`, `use_mlock=False`, `verbose=False` - `backend/services/llama_engine.py:103-119`, `backend/services/llama_engine.py:348-361`. KV cache is the temporary memory holding computed attention keys and values during generation, and it grows with context length.
- Threads default to probed physical cores - `backend/config.py:78-83`.
- GPU fallback. When `LLAMA_N_GPU_LAYERS` is None the load tries `n_gpu_layers=-1` then retries CPU `0` on exception with a warning. An explicit knob is used verbatim - `backend/services/llama_engine.py:120-128`, `backend/services/llama_engine.py:362-370`.
- Warmup probe. One max-token-1 non-streaming chat completion after load. Failure only warns - `backend/services/llama_engine.py:130-139`, `backend/services/llama_engine.py:372-381`. Embed has no GPU path and no warmup - `backend/services/llama_engine.py:191-216`.
- Embed load. `embedding=True`, ctx clipped to `min(EFFECTIVE_N_CTX, 2048)`, `n_batch=2048` and `n_ubatch=2048` so full-size chunks fit the encoder assert - `backend/services/llama_engine.py:201-214`.
- Larger `n_ctx` raises KV-cache and RAM cost, which is why the ctx bin gates the resident model choice - `backend/config.py:70-76`.

## 3. Resolution and fallback

Worker pick is pure path resolution and never loads weights - `backend/services/llama_engine.py:243-249`.

Priority order in `resolve_worker_model_path`:

1. Explicit knob `LLAMA_WORKER_MODEL`. Name lookup in the chat dir wins, else an existing file path wins - `backend/services/llama_engine.py:250-257`. Default knob value ships as `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` - `backend/config.py:39`.
2. Family plus size scan over chat-dir candidates. Small family (`lfm`, `qwen`, `smollm`, `ministral`) and small size (`1.2`, `1.5`, `1_5`, `1b`, `0.8b`, `2b`) first - `backend/services/llama_engine.py:239-240`, `backend/services/llama_engine.py:281-286`.
3. Family-only scan second - `backend/services/llama_engine.py:287-289`.
4. Chat fallback. `EFFECTIVE_CHAT_MODEL` with `is_fallback=True` - `backend/services/llama_engine.py:290`.

Exclusion rule. The active chat path is removed from worker candidates before scanning, so a worker-named chat model does not self-select - `backend/services/llama_engine.py:266-280`.

Summary slot choice in `resolve_slot` loads no weights. Explicit role wins, else a non-default `SUMMARY_MODEL_ROLE` is respected as-is, else auto-route to worker only when RAM is at or above `MIN_WORKER_RAM_GB` and the worker path resolves non-fallback - `backend/services/summarize_service.py:69-101`. RAM probe failure or resolve failure routes to chat - `backend/services/summarize_service.py:84-96`.

Model finder `list_models(kind, name)`. `name` picks one exact model, `kind` filters chat vs embed by `nomic` in the filename, most-recent-first otherwise - `backend/config.py:135-160`. Chat and embed resolved paths raise download-hint errors when missing - `backend/config.py:105-124`.

## 4. Bins and knobs

- `EFFECTIVE_N_CTX`. Explicit `LLAMA_N_CTX` wins, else 8192 when RAM is at or above 12.0 GB else 4096 - `backend/config.py:70-76`.
- `EFFECTIVE_N_THREADS`. Explicit `LLAMA_N_THREADS` wins, else probed physical cores - `backend/config.py:78-83`.
- `LLAMA_N_GPU_LAYERS`. None means auto Vulkan-or-CPU, else verbatim - `backend/config.py:42`, `backend/services/llama_engine.py:120-128`.
- Bin-dependent shares. Answer reserve, RAG share, history share, and memory tokens each have 4K and 8K values selected by the active ctx - `backend/config.py:85-103`. RAG means answering from uploaded files by retrieving relevant chunks first.
- Knobs. `LLAMA_MODEL_PATH`, `LLAMA_CHAT_MODEL`, `LLAMA_EMBED_MODEL`, `LLAMA_WORKER_MODEL`, `LLAMA_N_CTX`, `LLAMA_N_THREADS`, `LLAMA_N_GPU_LAYERS`, `SUMMARY_MODEL_ROLE`, `SUMMARY_TIMEOUT_S`, `MIN_WORKER_RAM_GB` - `backend/config.py:36-57`.

## 5. Runbook A - add a model slot

Use `WorkerEngine` as the worked example. Each step cites the line pattern to copy.

1. Subclass `LlamaEngine` in `backend/services/llama_engine.py`. Copy the `WorkerEngine` shell: role default, plus a fallback flag when the slot can fall back to chat - `backend/services/llama_engine.py:293-307`.
2. Override `load()` only. Follow the worker shape: return early when loaded, pick slot path when present else call your resolver, store `model_path`, build `common_kwargs` from `EFFECTIVE_N_CTX` and `EFFECTIVE_N_THREADS`, apply the `LLAMA_N_GPU_LAYERS` None-try-Vulkan-else-CPU branch, then run the 1-token warmup probe that only warns - `backend/services/llama_engine.py:327-381`. For an encoder slot copy `EmbeddingEngine.load` instead: `embedding=True`, clipped ctx, no GPU branch, no warmup - `backend/services/llama_engine.py:191-216`.
3. Add a pure `resolve_*` helper above the class. It must return paths only and never construct `Llama`. Copy `resolve_worker_model_path`: explicit knob first with `list_models("chat", name=...)` plus file-path fallback, then filtered scan, then chat fallback with an `is_fallback` bit - `backend/services/llama_engine.py:243-290`.
4. Register the slot in the shared registry. `LlamaEngine.get_instance` holds `cls._lock`, constructs the subclass for its role, and returns the same object per role - `backend/services/llama_engine.py:52-60`. Mirror `WorkerEngine.get_instance`: share `LlamaEngine._instances` and `LlamaEngine._lock`, construct on miss, repair a wrong-typed entry, delegate non-matching roles back to base - `backend/services/llama_engine.py:309-325`.
5. Assign ownership. Transient slots load on demand and unload in `finally` at the caller. Copy the worker pair `ensure_loaded` plus `finally: engine.unload()` - `backend/services/summarize_service.py:117-128`, `backend/services/rag_graph.py:77-86`. Copy the embed pair for graph-scoped use: construct with `EmbeddingEngine.get_instance("embed")` when owning, unload in the inner `finally` so unload runs even when graph invoke raises - `backend/services/rag_graph.py:243-247`, `backend/services/rag_graph.py:272-283`. Resident slots preload once in lifespan and unload only at shutdown - `backend/main.py:32-42`.
6. Route all generation through `LLMService`. Use the `invoke` shape `ensure_loaded`, `acquire`, try, finally `release` for blocking calls and the `astream_chat` shape for streams - `backend/services/llm_service.py:157-191`, `backend/services/llm_service.py:91-139`. Keep the never-nested rule: helpers that touch another slot run before any generation acquire - `backend/services/summarize_service.py:1-13`, `backend/services/rag_graph.py:418-420`.
7. Confirm discovery and health. `health()` reports status, model, path, exists, loaded, generating, and available from `list_models()` - `backend/services/llama_engine.py:152-174`. An empty model dir yields the `no-model.gguf` placeholder with status `error` - `backend/services/llama_engine.py:162-174`. `/api/health` maps `ready` to healthy - `backend/main.py:72-89`.

## 6. Runbook B - change RAM bins

Use this when moving a box between the 4K and 8K ctx bins or pinning threads.

1. Set the exact knob. `LLAMA_N_CTX` in `backend/config.py:40` is the ctx bin override. Leave it unset for auto bins from `EFFECTIVE_N_CTX` - `backend/config.py:70-76`. Set `LLAMA_N_THREADS` in `backend/config.py:41` only when pinning threads, else the probed core count applies - `backend/config.py:78-83`. Keep `MIN_WORKER_RAM_GB` in `backend/config.py:57` in mind: it gates worker auto-route in `backend/services/summarize_service.py:83-101`, not ctx directly.
2. Restart the backend process so `total_ram_gb()` and `physical_cores()` re-resolve once per process - `backend/config.py:16-28` - and lifespan reloads the chat slot - `backend/main.py:32-37`.
3. Confirm via the startup log line. A generative load logs `Model loaded - <name> (ctx=<n>)` - `backend/services/llama_engine.py:105`. A worker load logs `Worker model loaded - <name> (fallback=<bool> ctx=<n>)` - `backend/services/llama_engine.py:342-347`. An embed load logs `Embedding model loaded - <name>` - `backend/services/llama_engine.py:216`. Lifespan then logs `Chat Model loaded` - `backend/main.py:35`. Check that the logged `ctx` matches the intended bin: 4096 for below 12.0 GB or an explicit 4096 pin, 8192 for at or above 12.0 GB or an explicit 8192 pin - `backend/config.py:70-76`.
4. Confirm dependent shares followed the bin. Answer reserve, RAG share, history share, and memory tokens switch on `EFFECTIVE_N_CTX` - `backend/config.py:85-103`. If the log shows the wrong ctx, the shares are wrong too.

## 7. Rollback - pin version back

1. Pin the model file back. Set `LLAMA_CHAT_MODEL`, `LLAMA_WORKER_MODEL`, or `LLAMA_EMBED_MODEL` in `backend/config.py:37-39` to the prior GGUF filename. GGUF is the single-file model weight format loaded by llama.cpp. `list_models(kind, name=...)` resolves a bare name inside `LLAMA_MODEL_PATH` or an exact file path - `backend/config.py:135-147`.
2. Pin the bin back when the change was a bin edit. Restore the prior `LLAMA_N_CTX` value or unset it to return to auto bins - `backend/config.py:40`, `backend/config.py:70-76`.
3. Restart and re-check the same startup log lines from Runbook B: generative `Model loaded` with the old name and ctx - `backend/services/llama_engine.py:105` - plus `Chat Model loaded` - `backend/main.py:35`.
4. Check `/api/health`. Status `ready` with the pinned model and `loaded=true` means the rollback held - `backend/services/llama_engine.py:152-174`, `backend/main.py:72-89`.

## 8. Consumers - pointers only

This doc does not repeat consumer logic. Follow the pointer for the caller you are changing:

- User-facing chat turns, streaming, and Busy-to-429 mapping: docs/chat-pipeline.md.
- Retrieval, rewrite-before-acquire, and embed own-and-unload: docs/rag.md.
- Memory recall, episodic rollups, and summary slot routing: docs/memory-system.md. Rollup is the background job that summarizes older turns into episodic memory after a turn is saved.

## 9. Health and failure modes

- `health()` fields. Status, model, path, exists, loaded, generating, available - `backend/services/llama_engine.py:152-174`. `loaded` is handle non-null, `generating` is the single-flight flag - `backend/services/llama_engine.py:171-172`.
- Busy. `acquire` raises, HTTP maps to 429, summaries and rewrites skip or fall back instead of queuing - `backend/services/llama_engine.py:73-79`, `backend/services/summarize_service.py:176-178`, `backend/services/rag_graph.py:136-138`. Foreground paths check `is_generating()` before submitting work so they skip fast rather than contend - `backend/services/summarize_service.py:104-114`, `backend/services/rag_graph.py:66-74`.
- Missing weights. `ensure_loaded` raises normalized `RuntimeError`, `EFFECTIVE_*` raises `FileNotFoundError` with a download hint, `health` reports `error` - `backend/services/llama_engine.py:85-91`, `backend/config.py:105-124`, `backend/services/llama_engine.py:162-165`.
- GPU fallback. Vulkan failure retries on CPU - `backend/services/llama_engine.py:124-126`.
- OOM posture. Only one generative slot resident at a time: chat resident, worker transient. Embed unloads after each graph run. Batch sizes are fixed per slot - `backend/main.py:32-34`, `backend/services/rag_graph.py:272-283`, `backend/services/llama_engine.py:106-114`.

## 10. Test map

- Singleton per role plus lazy creation - `backend/tests/test_worker.py:46-55`.
- Worker resolution: fallback without weights, explicit knob wins, family-plus-size beats family-only, chat exclusion falls back - `backend/tests/test_worker.py:58-102`.
- Summary chat path: busy skip without invoke, resident slot kept with no unload, timeout falls back to extractive - `backend/tests/test_worker.py:105-141`.
- Summary worker path: unloads after success and after failure - `backend/tests/test_worker.py:144-166`.
- Auto-route matrix of RAM by worker-file by explicit role - `backend/tests/test_worker.py:169-257`.
- Degenerate-output screening and invoke kwargs on both paths - `backend/tests/test_worker.py:260-350`.
- RAG tests use fake engines for graph and ingestion paths - `backend/tests/test_rag.py:47`, `backend/tests/test_rag.py:533`.
- Episodic tests patch the chat singleton for busy and idle cases - `backend/tests/test_episodic.py:191`, `backend/tests/test_episodic.py:206`.
