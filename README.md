# Everything RAG — Your Personal Second Brain

**English | [简体中文](README.zh-CN.md)**

**Local-first RAG (Retrieval-Augmented Generation) Q&A** that chunks and indexes your own Markdown documents (including AI conversations exported as Markdown) and answers natural-language questions, citing traceable sources whenever there is local or retrieved evidence. Knowledge files and runtime state stay on your machine; chat, embedding, image-understanding and web-search requests go to whichever external services you configure.

The goal is a RAG Agent with MCP, Skills and Web Search support, plus solid chunking and recall. It is being built one step at a time.

## Features

- **Local knowledge base management**: import a Markdown directory (nested subdirectories included), vectorize it in one click, and get answers with source citations wherever possible.
- **Incremental sync**: fingerprint comparison (a fast mtime + size path, confirmed by content hash) classifies files as added / updated / deleted / unchanged, so editing a source later only re-indexes what changed instead of re-chunking everything.
- **Vision understanding (images into the index)**: when image understanding is configured, pictures in documents produce a standalone retrieval summary plus detailed evidence. The summary, file name and headings feed vector retrieval; table rows and columns, numbers and flow relationships stay in the evidence and also feed keyword retrieval. Descriptions are cached and reused across documents.
  - Images are recognized at higher resolution and the originals are cached locally. Questions about images, tables or numbers verify at most two recalled images in parallel, replacing stale descriptions with this run's reading; failures are explicitly marked as unverified. The source panel supports tables and "view original image". Ordinary questions do not call the vision model.
  - v7 stores image summary vectors and evidence separately. An existing v6 index can be migrated while imports are idle by running `backend/.venv/Scripts/python scripts/migrate_image_index.py`: text vectors are reused and only image vectors are rebuilt; add `--promote` after the evaluation passes to update the index manifest. Old collections and state backups are kept. Migration reuses historical descriptions rather than re-recognizing every old image, and old originals are cached on demand during verification. Re-recognition and on-demand verification incur provider charges.
- **Markdown-oriented chunking**: tuned for the common problems in Yuque / Obsidian exports.
- **Hybrid retrieval + precise recall**: dense vectors + BM25 lexical (with heading injection) + RRF fusion + multi-stage relevance gating.
- **Web search**: optional Tavily web search while answering.
- **Flexible API configuration**: chat, embedding, vision and web search are all configurable.

## Why the chunking and retrieval work

### Markdown chunking (for messy source documents)

| Technique | Problem it solves |
|---|---|
| Heading-level chunking + short-paragraph merging | orphan fragments caused by blank lines / truncation; step-by-step documents split across images |
| Table semantics (header → `field: value`, rowspan inheritance) | merged-cell tables flattened into noise |
| Mixed HTML cleanup + code protection | removes presentational tags such as `<font>` / `<span>` and invisible scripts while keeping code fences, inline code, XML and generics |
| `plain` / `text` pseudo-fences treated as prose, indentation not mistaken for code | Yuque exports label prose as code |
| Low-information noise filtering (tags / URLs / punctuation only) | less dense noise while keeping pricing, statistics and other numeric tables |
| Heading values preserved | facts that live only in an empty subheading (URLs, environment requirements) are no longer lost |
| Semantic segmentation (split where adjacent-paragraph embedding similarity drops sharply) | boundary drift when a single heading covers several topics |
| Image summary index + separate evidence + on-demand original verification | avoids long OCR diluting semantics while keeping numbers, tables and traceable originals |

### Retrieval

- **Hybrid retrieval**: dense (Qwen3-Embedding-8B, cloud) + BM25 lexical (bigram + file-name / heading injection) + RRF fusion.
- **Multi-stage relevance gating**: base gate 0.60 + lexical rescue 0.45 (short abbreviations / heading hits) + a penalty for pure-dense results with no lexical match + a penalty for numeric noise.
- **Parent-child retrieval**: a matched chunk expands to its parent section (the enclosing heading) context, with one context slot reserved for the true parent section — instead of dragging in a whole document.
- **Measured precision (fresh isolated v5 index)**: across 66 strict chunk-level positives (file basename + full heading + body evidence), **Hit@1 = 78.79%, Hit@5 = 96.97%, Hit@8 = 100%, MRR@8 = 85.85%**; the out-of-corpus false-admission rate over 10 negatives was **0**. 202/202 documents parsed, 2791/2791 vectors reachable by ANN.

## In action

![Screenshot](./docs/001.png)

![Screenshot](./docs/002.png)

## Model costs

![API bill](./docs/004.png)

![API bill](./docs/005.png)

The bills above: 1,630 images processed with **Qwen/Qwen3-VL-32B-Instruct** for CNY 7.46; a knowledge base of roughly **500k characters** chunked into 3,198 chunks and embedded with **Qwen/Qwen3-Embedding-8B** for CNY 1.5 — CNY 9.1 in total. Chat model costs are omitted here.

All in all, a fairly cheap setup.

## Quick start

### 1. Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11+ | backend |
| Node | 18+ | only needed to build the frontend |

> Embedding and vision both go through cloud OpenAI-compatible APIs; no large local model download is required.

### 2. Install dependencies

```bash
# Backend (Windows: backend\.venv\Scripts\python)
cd backend
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

# Frontend (optional — the backend serves the existing dist/ as-is)
cd ../frontend
npm install
```

### 3. Configure models (chat / embedding required, vision / search optional)

Pick either option; the **Settings page** is recommended (graphical, takes effect immediately):

- **Option A (recommended)**: after launching, fill in "chat model / embedding model / vision model / web search / answer preferences" on the Web settings page. RAG identity, grounding, prompt-injection defenses and citation rules are fixed in code and cannot be overridden from the page.
- **Option B (environment variables)**: copy the template `backend/.env.example` to `backend/.env` and fill in the variables below.

| Variable | Required | Description |
|---|---|---|
| `EVERYTHING_RAG_CHAT_BASE_URL` / `_MODEL` / `_API_KEY` | ✅ | Chat model (OpenAI-compatible endpoint) |
| `EVERYTHING_RAG_EMBED_BASE_URL` / `_MODEL` / `_API_KEY` | ✅ | Embedding model (OpenAI-compatible `/embeddings`) |
| `EVERYTHING_RAG_SEARCH_API_KEY` | ✳ | Web search (Tavily, optional) |
| `EVERYTHING_RAG_DATA_DIR` | ✳ | Data directory, defaults to `~/.everything-rag` |

### 4. Launch

```bash
# Back in the repository root
backend/.venv/Scripts/python scripts/run.py              # fixed port 9999 + opens the browser
backend/.venv/Scripts/python scripts/run.py --no-browser # for automation / CI (prints the port only)

# Or double-click EverythingRagStart.bat (Windows)
```

Then: **configure models on the Settings page → import a Markdown directory → incremental sync → start asking questions**.

## Common commands

```bash
# Full import from the command line (same as the UI import, including vision; image descriptions are filled in the background)
backend/.venv/Scripts/python scripts/import_md.py --dir "<your document directory>"

# Backend tests (Windows)
cd backend && .venv/Scripts/python -m pytest

# Frontend build
cd frontend && npm run build

# E2E (Playwright starts an isolated frontend test server)
cd frontend && npm test
```

## V1.2.0: Agent orchestration and workspace isolation

V1.2.0 upgrades the Q&A pipeline from "single retrieval + single generation" to a **controlled multi-Agent orchestration**, and turns "data never leaves the machine, answers always carry sources" into verifiable mechanisms.

- **Workspace isolation**: each workspace owns its own directory tree (`documents/` / `chroma/` / `state.db` / `app.db` / `memory.json` / `checkpoints.db`). Dense search, BM25, section supplementation, browsing, statistics and image / task identity **only read the current workspace**; a cross-workspace image hash or task ID always returns 404.
- **Agent orchestration (LangGraph)**: intent routing → plan (multiple tasks) → parallel research → synthesis → verification → publication. The research subgraph has round and stagnation limits, and when verification fails the system publishes an **answer with explicit gaps** rather than inventing conclusions.
- **RunManager and SSE**: `POST /api/workspaces/{id}/runs` creates a task (**202**), and `GET .../runs/{id}/events` subscribes to v2 events (`schema_version=2`, 20 event types), with `after=` / `Last-Event-ID` replay, cancel and resume. The old `POST /api/chat` entry point is retired (**410**, with a pointer to the replacement).
- **Model capabilities and tool protocol**: capabilities (context window / max output / tools / thinking tier) are resolved from `base_url + model`; DeepSeek's official `deepseek-flash` is in the table. Four thinking levels (off / light / deep / maximum, **off by default**) map to real request parameters; multi-turn tool calls return the full protocol history and usage.
- **Answer preferences**: the preferences and custom prompt (role / tone / length) from the settings page **take effect in every answer-writing path of the run pipeline** (General Agent answers + synthesis-node candidate answers) via the `system` responsibility prompt; on conflict, responsibility and immutable rules win.
- **Permissions and approvals**: three permission modes (ask for approval / auto-approve / full access); web tools pause and raise an approval card before executing. Repeated approvals are idempotent, an old decision is rejected (409) after its arguments or context change, and private names are redacted **before the request actually leaves the machine**.
- **Workspace memory**: `memory.json` uses atomic writes + optimistic concurrency + `operation_id` idempotency; the snapshot is frozen when a session is created (session A's snapshot does not change when memory is later updated); manual deletions / corrections / closures take effect in the context of the next model request; Chinese history search runs on FTS5 + jieba.
- **Frontend workflow**: a process area (collapsible timeline + expandable research tasks), approval cards (allow once / allow this kind for this run / deny + edit query), permission and thinking menus, and a memory panel. After a disconnect or refresh, missed events are replayed by `event_seq`, and answer fragments arrive before the completion event.

Acceptance and evaluation: `docs/tech/v1.2.0-acceptance-report.md` (A01–A32 item by item), `v1.2.0-quality-report.md` (routing / answer quality), `v1.2.0-migration-report.md` (migration and keep / exclude list).

**Verification status (measured 2026-09-20, numbers only)**

- Acceptance suite: `backend/tests/test_acceptance_v120.py` (A01–A32 + checkpoint interrupt / resume) + `test_acceptance_t27_followups.py` (6 endpoint-level + 3 assembly-level) = **42 passed / 4 skipped (live) / 0 failed / 0 xfailed**.
- Regression: full backend run `2 failed / 1719 passed / 86 skipped` (the 2 failures are always the local fixed-port baseline `test_run_py.py`; all 86 skips are live gates: 4 A items + 82 real-index recall gates); `ruff check app tests` is clean; the frontend `npm run build` has zero errors and Playwright reports **75 passed**.
- The routing evaluation M1–M6 fully meets targets (155 cases = 87 real + 68 constructed, reproducible artifacts).
- **Not met / not executed (no sugar-coating)**: session moves (A27) **meet the target at the backend endpoint level but not on the real frontend path** (the frontend does not send `confirm` / `expected_context_epoch` / `origin_run_id`, so a session carrying history receives a 409 in the UI; separately, the legacy drag-and-drop move entry bypasses the state machine). The four live items (real news, real four-level Q&A, real multi-turn tool calls, real data migration) and the PRD 16.4 answer-quality / thinking-level evaluations are **not executed** (pending user authorization, see `docs/tech/v1.2.0-live-eval-plan.md`). Enable with: `EVERYTHING_RAG_LIVE_EVAL=1` (A tier) / `EVERYTHING_RAG_RUN_LIVE_EVAL=1` (real-index recall gates).

## Roadmap

* MCP and Skills support
* Support for more data sources
* More accurate chunking strategies
* ...
