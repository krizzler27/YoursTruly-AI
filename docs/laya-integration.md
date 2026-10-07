# Laya Decision Model: Retrain, Evaluate, and Ship

> Laya is the small decision model that answers plumbing questions for each chat turn: which capability should serve the message, whether personal memory is needed, how relevant each retrieved chunk is, whether a follow-up needs rewriting, and whether a draft answer is supported by its context. RAG means answering from uploaded files by retrieving relevant chunks first, and it is one of the routes Laya picks. This doc is the single reference for its knobs, data rules, training runbook, evaluation, and checkpoint delivery. Runtime chat wiring is not duplicated here - see `docs/chat-pipeline.md` for per-turn flow and fallbacks.

## 1. Verify in 5 minutes

Run these from the repo root. All paths below are repo-root-relative.

### 1.1 Smoke-test a checkpoint

```bash
python backend/notebooks/laya_dataset/test_finetuned.py --model-dir backend/laya-model --device cpu
```

- Source: `backend/notebooks/laya_dataset/test_finetuned.py:101-118` (flags `--model-dir`, `--device` at `backend/notebooks/laya_dataset/test_finetuned.py:103-104`).
- What it does: 7 cases on phrasings absent from training, one per knob, printing `PASS`/`FAIL` with confidence and latency per case (`backend/notebooks/laya_dataset/test_finetuned.py:110-118`). Exit 0 only on 7/7.
- Bar: routing and memory decisions must clear 0.8 confidence. Persistent hedging at ~0.3-0.5 after correct labels indicates a boundary-data deficit or a temperature problem, and blocks confidence gating for that head.
- Last measured on CPU (Oct 2026): 7/7 PASS at ~660-930 ms per case. Route RAG 0.52 (soft, safe-defaults cover it), route DIRECT correct at 0.18, hit exact 0.85, hit irrelevant 0.84, rewrite 0.835, grounding-hallucinated correctly low 0.143, memory 0.815. A safe-default is the fail-open fallback taken when confidence is low or inference errors, for example keeping fused order or the original query. Re-run the command above to refresh these numbers after any retrain.

### 1.2 Check dataset counts match the report

```bash
python -c "import json, collections; rows=[json.loads(l) for l in open('backend/notebooks/laya_dataset/laya_synthetic_dataset_v3.jsonl', encoding='utf-8') if l.strip()]; print('cases:', len(rows)); print(dict(collections.Counter(r.get('workflow') for r in rows)))"
python -c "import json; rows=[json.loads(l) for l in open('backend/notebooks/laya_dataset/laya_synthetic_dataset_v3.jsonl', encoding='utf-8') if l.strip()]; print('decisions:', sum(len(r['questions'] if isinstance(r['questions'], dict) else json.loads(r['questions'])) for r in rows))"
python -c "import json; print(json.load(open('backend/notebooks/laya_dataset/laya_synthetic_dataset_v3_report.json')))"
```

- Expected: 1869 cases, 2663 decisions, `{"rewrite": 200, "routing": 660, "retrieval_grade": 400, "memory": 309, "grounding": 300}` per `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3_report.json:1-12`.
- If the `.jsonl` and `_report.json` disagree, the `.jsonl` plus this check wins and the report file needs regenerating before any GPU run.

### 1.3 Check checkpoint layout

```bash
python -c "from pathlib import Path; d = Path('backend/laya-model'); print(d.resolve()); print(sorted(p.name for p in d.iterdir()))"
```

- Required entries: `rl_agent_config.json`, `model.safetensors`, `tokenizer/`, `encoder/`. The loader requires the whole directory, not weights alone, loaded via `laya.Agent(model_dir, device=...)` in `backend/services/laya_service.py:94-98`.
- Default location is `LAYA_MODEL_DIR` in `backend/config.py:58`. The dir is git-ignored and never committed. A missing dir raises with a train-or-set-pointer message (`backend/services/laya_service.py:88-92`).

### 1.4 Check knob thresholds

All gates are nominal 0.8. Defined in `backend/config.py:58-63`:

| Name | Default | Meaning |
| --- | --- | --- |
| `LAYA_MODEL_DIR` | `backend/laya-model` | Fine-tuned weights dir, repo-root-relative unless absolute - `backend/config.py:58`, resolved in `backend/services/laya_service.py:71-78` |
| `LAYA_ROUTE_THRESHOLD` | `0.8` | Route confidence gate - `backend/config.py:59` |
| `LAYA_MEMORY_THRESHOLD` | `0.8` | Needs-memory confidence gate - `backend/config.py:60` |
| `LAYA_HIT_THRESHOLD` | `0.8` | Hit-grade confidence gate, below keeps fused order (advisory) - `backend/config.py:61` |
| `LAYA_GROUND_THRESHOLD` | `0.8` | Is-grounded confidence gate, below substitutes the refusal - `backend/config.py:62` |
| `LAYA_REWRITE_THRESHOLD` | `0.8` | Rewrite-needed confidence gate, below keeps original query - `backend/config.py:63` |

- See section 6 for the confidence bar and section 7 for runtime pointer. Do not lower a threshold to make the smoke test pass; fix data or temperature instead.

## 2. Retrain runbook

Executable spec is `backend/notebooks/laya_finetune.ipynb` (Kaggle, GPU T4 x2, internet on). Training follows the upstream Laya RLCD recipe, unchanged in mechanics, applied to our data.

### 2.1 Prerequisites

- Kaggle notebook settings: Accelerator `GPU T4 x2`, Internet `On`. Checkpoints and final model save to `/kaggle/working/laya_finetuned_yours`.
- Upload `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3.jsonl` as Kaggle input data (cell 3 auto-resolves under `/kaggle/input/`; local GPU box path is `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3.jsonl`).
- Run cells top to bottom. Do not skip temperature fitting.

### 2.2 Procedure with exact cells

1. **Cells 1-2: environment and install.** Run the GPU check (`!nvidia-smi`, assert 2 GPUs) then the pip cell:
   ```bash
   pip install -q -U "laya>=0.1.6" "transformers>=4.48.0" "datasets>=3.0.0" safetensors huggingface_hub pyarrow pandas scipy accelerate tabulate
   ```
2. **Cell 3: preprocess.** Converts dataset rows to tokenized training items (`/kaggle/working/train_items.pt`) and a family-aware 80/20 held-out split (`/kaggle/working/test_cases.json`). The cell prints `kept / marker-drop / key-unmatched` counters per (workflow, question). Gate: every workflow must appear in `kept`; anything elsewhere is investigated before training.
3. **Cell 4: write `train_ddp.py`.** Writes the DDP GRPO-style policy-gradient script (proper-scoring rewards plus soft cross-entropy guidance, 4 epochs, per-group learning rates encoder 2.5e-5 and heads 1e-4) to `/kaggle/working/train_ddp.py`. A rolling checkpoint is written per epoch, so interrupted runs resume at most one epoch back.
4. **Cell 5: launch training.** Runs both T4 GPUs in parallel (~4 to 6 minutes):
   ```bash
   torchrun --standalone --nproc_per_node=2 /kaggle/working/train_ddp.py <MODEL_DIR> /kaggle/working/laya_finetuned_yours
   ```
   `MODEL_DIR` is the base-checkpoint dir from cell 3; output dir is `/kaggle/working/laya_finetuned_yours`.
5. **Temperature fitting (end of training script, automatic).** Per-type temperatures are refit on items held out of training. Calibration means stated confidence matches real accuracy, so gates can trust the numbers. This step is load-bearing for calibration; skipping it ships an overconfident model.
6. **Cells 6-7: held-out evaluation.** Loads `/kaggle/working/test_cases.json` and runs the fine-tuned agent over the split, computing accuracy, soft accuracy, Brier score, ECE, score MAE, and per-workflow accuracy plus the head-to-head comparison table.
7. **Cell 9: persist the benchmark report.** Writes the full metrics plus per-workflow breakdown to `/kaggle/working/laya_yours_benchmark_report.json` and to `<OUTPUT_DIR>/benchmark_report.json` next to the weights. (Cell 8 is optional Hugging Face Hub push only.)

### 2.3 After training

- Download `/kaggle/working/laya_finetuned_yours` to `backend/laya-model/` (or set `LAYA_MODEL_DIR` to its absolute path).
- Re-run sections 1.1 and 1.3 against the new dir before using it in the app.
- If training was interrupted, resume from the rolling `checkpoint_latest/` dir (one epoch back at most); do not mix temperatures across runs.

## 3. Dataset specification

Current dataset: `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3.jsonl` (1869 cases, 2663 decisions). Minimum viable scale is ~1000 cases / ~2000 decisions with at least ~100 rows per class.

### 3.1 Row schema

One JSON object per line: `{workflow, state, questions, gold}`.

- `state` is exactly what the knob observes at runtime. Routing: query, conversation history, attached-docs inventory. Grading: query, chunk text, filename. Grounding: query, context, answer.
- `questions` uses frozen templates, byte-identical on every row (see 3.3). `choice` questions carry at most ~20 options; option text shares a fixed token budget (`head_max_len`), so large label sets degrade sharply.
- `gold` carries soft probability distributions summing to 1.0, with the top probability varied in [0.75, 0.92] and no 0.0/1.0 entries. Hard labels destroy calibration and are forbidden. Single-question rows may store gold flat (`{label, probabilities}`); multi-question rows wrap per question id. Training code normalizes both shapes.

### 3.2 Workflow catalog

| Workflow | Questions | Rows | Function |
| --- | --- | --- | --- |
| routing | route (choice) + needs-docs (noul) [+ needs-memory] | 660 | select serving capability per message |
| retrieval_grade | hit_grade (score 0/1/2) | 400 | score each retrieved chunk; rank and truncate by score |
| rewrite | rewrite_needed (noul) | 200 | detect vague follow-ups needing query rewriting |
| grounding | is_grounded (noul) | 300 | detect answers unsupported by context |
| memory | needs_memory (noul) | 309 | detect dependence on personal memory |

Counts verified by the section 1.2 commands against `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3_report.json:1-12`.

### 3.3 Template discipline

Question instructions and criteria are part of the model input, i.e. they are prompts. Templates are therefore frozen: identical wording on every row, in training and in serving. Changing serving wording without retraining (or at minimum re-evaluation) is a behavior change and is treated as one. Wording improvements are tested training-free first by running the smoke test with variant templates against current weights; adopted variants are locked in and become the training templates of the next round.

### 3.4 Diversity requirements (normative)

The following rules are mandatory for any training batch. Each encodes a measured failure from rounds 1-3:

1. **Unique queries.** Every query unique after lowercasing. Numbered suffixes (`Ref #21`, `case #5`) counterfeit uniqueness and teach template memorization. Round 1 shipped 1599 rows with ~812 real patterns.
2. **Label-by-context pairing.** Every label must occur in every context state it can meet in production. DIRECT queries must sometimes carry a non-empty inventory; RAG queries sometimes an empty one; memory-negative queries must sometimes sit over name-bearing histories. Violations train shortcuts (round 2: "inventory listed implies RAG", routing at coin flip).
3. **Contrastive coverage on boundaries.** For each gate, include near-identical pairs with opposite labels: same query with and without the relevant doc; same history with a memory-needing versus a factual query (round 3).
4. **No repeated payloads.** No chunk, context, or history text repeated more than ~3 times across the set.
5. **Generic/project mix.** Roughly 60% everyday queries to 40% project-specific ones, so behavior holds outside our docs.
6. **Merge-time checklist.** Before any GPU run: per-workflow counts, unique-query ratio, and label-by-context tables.

### 3.5 Split protocol

Splits are family-aware, never random by row. Rows group by (workflow, normalized query with numbers masked; chunk text for grading; context text for grounding), and whole families go to train or test. Random splits leak templates across the split and report memorization as accuracy (observed: 0.996 held-out accuracy against hedging fresh probes). Contrastive pairs intentionally share a family so a pair never splits.

Implemented in cell 3 of `backend/notebooks/laya_finetune.ipynb` (seed 42, 80/20 by family; calibration slice seed 20260922 held out again inside `train_ddp.py` before sharding).

## 4. Evaluation protocol

Two tiers, both required:

1. **Held-out metrics (notebook cells 6-7, 9).** Accuracy, soft accuracy, Brier, ECE, score MAE, per-workflow accuracy, plus `benchmark_report.json` next to the weights. Interpret with the split protocol in mind: high argmax accuracy with weak soft accuracy or score MAE indicates memorized labels over uncalibrated distributions.
2. **Fresh-probe smoke test (`backend/notebooks/laya_dataset/test_finetuned.py`).** Seven cases on phrasings absent from training, one per knob, reporting PASS/FAIL with confidence and latency. This is the honest signal for generalization. Routing and memory decisions must clear 0.8 confidence; persistent hedging at ~0.3-0.5 after correct labels indicates a boundary-data deficit (round 3) or a temperature problem, and blocks confidence gating for that head.

Run tier 2 with the section 1.1 command after every retrain. Do not ship a checkpoint that fails tier 2 even if tier 1 looks strong.

## 5. Checkpoint delivery

- Location: `backend/laya-model/` (default `LAYA_MODEL_DIR` - `backend/config.py:58`, git-ignored, never committed). Verify layout with the section 1.3 command: `rl_agent_config.json`, `model.safetensors`, `tokenizer/`, `encoder/`.
- Missing dir raises with a train-or-set-pointer message - `backend/services/laya_service.py:88-92`.
- The `rl_agent_config.json` carries the fitted per-type `temperature` array; inherited bucket overrides are removed at save time so the new values are not hidden.
- Copy the notebook `benchmark_report.json` next to the weights when shipping; it is the audit trail for the numbers claimed in section 1.1.

## 6. Runtime wiring (pointer plus wrapper contract)

Full per-turn flow, decision points, failure modes, and test map: `docs/chat-pipeline.md`. Related budgets: `docs/memory-system.md` for memory interaction, `docs/rag.md` for retrieval use of graded hits. This section keeps only what must not break: the wrapper contract, the knob table (section 1.4), and the frozen-template rule.

### 6.1 Wrapper lifecycle

Lifecycle in `backend/services/laya_service.py`:

- Singleton holds no weights - `get_instance` creates empty, never loads - `backend/services/laya_service.py:55-60`.
- Lazy load on first `predict` - `load` checks `is_loaded`, resolves dir, constructs `laya.Agent(str(model_dir), device="cpu")` - `backend/services/laya_service.py:83-102`.
- CPU-only - keeps Vulkan for the chat slot, checkpoint verified on CPU at ~700-930 ms per case - `backend/services/laya_service.py:96-98`. A slot is one model holder with one role where only one generation runs at a time.
- `predict(state, questions)` is one transient pass - `load`, `agent.predict`, `unload` in `finally` - `backend/services/laya_service.py:116-124`.
- `predict_many(states, questions)` batches under one load - one load serves a hit list, `unload` in `finally`, same never-resident contract - `backend/services/laya_service.py:126-138`.
- `unload` deletes the agent and runs `gc.collect`, mirroring `EmbeddingEngine` - `backend/services/laya_service.py:104-114`.
- Never preloaded in main lifespan, never co-resident with chat weights - `backend/services/laya_service.py:1-7`.
- Load failure raises `RuntimeError` once with a clear message - missing dir vs load exception - `backend/services/laya_service.py:88-101`. Callers treat it as fail-open, never crash the turn.
- `reset_instance` is test-only - `backend/services/laya_service.py:62-69`.

Laya serves through one lazy singleton: load, predict, unload per use, never resident, any error fails open to the SLM path. Chat plumbing needs typed decisions - route, grade, rewrite, ground, recall - without paying text-generation cost or parse risk on the 8 GB box. Weights must never crowd out the chat slot and every gate must fail open to today's behavior when confidence is low or inference errors. Fail-open means falling back to the safe-default instead of crashing the turn.

### 6.2 Frozen-template operational rule

Question wording is model input: byte-identical on every training row and every serving call. Never reword without retraining (or at minimum re-evaluation) - section 3.3, `backend/services/laya_service.py:19-21`.

Quoted byte-identical from the smoke test `backend/notebooks/laya_dataset/test_finetuned.py:13-84`:

- Route (`choice`) - `backend/notebooks/laya_dataset/test_finetuned.py:19-22`, served from `backend/services/laya_service.py:22-30`:
  - instructions: `Which capability should serve this message?`
  - `DIRECT`: `greetings, smalltalk, general knowledge answerable without files`
  - `RAG`: `needs attached local files or ingested documents`
  - `WEB`: `needs fresh internet info, current events, latest versions`
- Needs-memory (`noul`) - `backend/notebooks/laya_dataset/test_finetuned.py:80-81`, served from `backend/services/laya_service.py:32-35`:
  - instructions: `Does this need personal memory like name, preferences, or earlier conversation?`
- Hit-grade (`score`) - `backend/notebooks/laya_dataset/test_finetuned.py:42-43`, served from `backend/services/laya_service.py:37-41`:
  - instructions: `How relevant is this chunk to the query?`
  - criteria: `["irrelevant", "partial", "exact"]`
- Rewrite-needed (`noul`) - `backend/notebooks/laya_dataset/test_finetuned.py:61-62`:
  - instructions: `Does this query need rewriting to be self-contained for retrieval?`
- Is-grounded (`noul`) - `backend/notebooks/laya_dataset/test_finetuned.py:71-72`:
  - instructions: `Is this answer fully supported by the provided context, with no outside facts?`

Wording improvements are tested training-free first by running the smoke test with variant templates against current weights; adopted variants lock in as the next round's training templates - section 3.3.

## 7. Background (read after the runbook)

### 7.1 System One decision models

Large chat models generate text token by token. For agentic plumbing (route this query, grade this chunk, check this answer) text generation is the wrong interface: outputs must be parsed, may hallucinate, and cost seconds on small hardware. System One models (TypeSafe Jev; the open-weight Laya family by Convai Innovations, Apache-2.0) take the opposite trade: a single encoder forward pass over structured inputs returns typed decisions with calibrated probabilities. No tokens are generated, so there is nothing to parse and no surface for hallucination in the decision itself.

### 7.2 Decision primitives

Laya exposes exactly three output types. Every knob in this project is exactly one of them:

| Primitive | Output | Used for |
| --- | --- | --- |
| `choice` | winning label, per-option probabilities, confidence | route selection over DIRECT / RAG / WEB |
| `noul` | P(true) in [0, 1] | binary gates: needs-docs, needs-memory, rewrite-needed, is-grounded |
| `score` | expected level on an ordinal rubric plus distribution | rankers: chunk relevance (irrelevant / partial / exact) |

### 7.3 Calibration

Laya is trained with RLCD (reinforcement learning against strictly proper scoring rules), which rewards honest uncertainty rather than bold guesses. After fine-tuning, one temperature parameter per question type is refit on held-out data so that stated confidence matches empirical accuracy (expected calibration error typically 0.08-0.15 after fitting, versus ~0.3-0.5 before). Confidence gating in production depends on this property; an uncalibrated head must not gate actions.

### 7.4 Capacity and latency envelope

Reference checkpoints are 322M (multilingual) to 421M (English) parameters: ~0.3-0.8 GB in FP16/INT8, versus ~2.2 GB for the 3B chat model. Measured single-question latency is ~33 ms on a T4 GPU and 200-900 ms on CPU. Throughput rises sharply under batching because all questions in one call share a single forward pass.

### 7.5 Why fine-tuning is required

The released base checkpoints score near chance on our tasks (~0.36 accuracy against a 0.32 random baseline). All task capability comes from specialization: fine-tuning adapts the shared encoder and decision heads to our question templates and our data distribution. A general decision model does not exist here; there is only a base to specialize.

## 8. Revision history

- **v1 (agent-generated, 1599 rows).** Correct schema, failed diversity: ~812 real patterns, DIRECT/RAG perfectly correlated with inventory presence.
- **Round 2 (+150 rows).** 100 DIRECT-with-inventory queries, 50 memory hard negatives. Fixed the inventory shortcut directionally; routing confidence still hedging.
- **v3 (+120 contrastive rows; 1869 cases, 2663 decisions).** Same-query opposite-label pairs for routing and memory, plus the family-aware split. Smoke test 7/7; routing confidence remains the open item for round 4.

## 9. File index

- `backend/notebooks/laya_finetune.ipynb` - training notebook (cells 3-9 as specified in section 2).
- `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3.jsonl` - training data.
- `backend/notebooks/laya_dataset/laya_synthetic_dataset_v3_report.json` - counts and provenance.
- `backend/notebooks/laya_dataset/test_finetuned.py` - smoke test.
