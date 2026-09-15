<div align="center">

# AVA — Artificial Virtual Assistant

*A production-grade, voice-first AI assistant with cognitive memory architecture*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C?style=flat-square)](https://langchain.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=flat-square)](https://groq.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google)

</div>

---

## What is Ava?

Ava is a **voice-first personal AI assistant** built from the ground up — designed to feel like talking to someone who genuinely *remembers* you: your preferences, your projects, your unfinished conversations.

Unlike off-the-shelf chatbots that reset with every new tab, Ava maintains a **3-layer cognitive memory architecture** inspired by human memory research: a short-term working buffer, a semantic long-term fact store, and an episodic autobiographical record. Pair that with sub-35ms voice response optimization, real-time web search grounding, and true multi-user isolation — and Ava becomes something far more personal than a prompt wrapper.

This README documents the engineering journey: the technology we chose, the hard problems we ran into, and the production-quality solutions we built.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [How We Built It](#how-we-built-it)
4. [Major Challenges & Solutions](#major-challenges--solutions)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Running Tests](#running-tests)

---

## Architecture Overview

```
User
 │
 │  Voice (PCM16 @ 16kHz)  /  Text
 ▼
┌──────────────────────────────────────────┐
│             STREAMLIT UI                 │
│  ChatGPT-style sidebar  │  WebRTC Mic   │
│  Session switcher       │  st.audio_input│
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│           VOICE PIPELINE                 │
│  Sarvam AI Realtime WebSocket STT        │
│  Groq Whisper  (quick voice notes)       │
│  Sarvam AI TTS (spoken responses)        │
└──────────────────────────────────────────┘
               │  Transcript
               ▼
┌──────────────────────────────────────────┐
│         LANGCHAIN AGENT CORE             │
│  LangChainResilientAgent                 │
│  Tools: task │ diary │ web_search │ cal  │
└──────────────────────────────────────────┘
       │                      │
       │ Context Inject        │ Post-Turn Distill
       ▼                      ▼
┌──────────────────────────────────────────┐
│   3-LAYER COGNITIVE MEMORY               │
│                                          │
│  Short-Term    Semantic      Episodic    │
│  Sliding Win   Fact Vectors  Episodes   │
│  Eviction LLM  Gemini Embed  pgvector   │
│                Cosine Search 3072-dim   │
│                                          │
│  MemoryManager  │  Context Bridging      │
└──────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│         POSTGRESQL  (pgvector)           │
│  chat_messages   conversation_summaries  │
│  episodes        semantic_facts          │
│  tasks           diary_entries           │
│  Full multi-tenant isolation by user_id  │
└──────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | ChatGPT-style sidebar, chat bubbles, dual voice modes |
| **Agent** | LangChain | Tool-use loop, resilient agent with retry |
| **LLM** | Groq (`openai/gpt-oss-120b`) | Primary inference, fast token generation |
| **Embeddings** | Google Gemini (`text-embedding-004`) | 3072-dim semantic & episodic vectors |
| **STT** | Sarvam AI Realtime WebSocket | Streaming speech-to-text |
| **STT (Quick)** | Groq Whisper | 1-click voice note transcription |
| **TTS** | Sarvam AI | Voice response synthesis |
| **WebRTC** | `streamlit-webrtc` + PyAV | Browser microphone, 48 → 16kHz resampling |
| **Database** | PostgreSQL + `pgvector` | Persistent storage, vector similarity search |
| **Web Search** | Firecrawl | Real-time web grounding for factual queries |
| **Config** | Pydantic Settings | Typed environment configuration |
| **Runtime** | `uv` | Fast dependency management |

---

## How We Built It

### Phase 1 — Foundation: Agent + Database

Started with a minimal LangChain agent connected to PostgreSQL. The first working version managed tasks and diary entries through natural language, with tools wired to a clean SQLAlchemy ORM layer. We chose PostgreSQL from day one — not SQLite — to ensure production readiness and enable `pgvector` for semantic search from the start.

---

### Phase 2 — Voice Pipeline

Adding voice required wiring together three services: **Sarvam AI Realtime WebSocket** for streaming STT, **Groq Whisper** for quick-note transcription, and **Sarvam TTS** for spoken responses. The WebRTC integration (`streamlit-webrtc`) captures microphone frames in the browser and streams them to the backend, while PyAV handles the 48kHz → 16kHz audio resampling that Sarvam requires.

---

### Phase 3 — Cognitive Memory Architecture

The core engineering milestone. We implemented a 3-layer memory system modelled after cognitive science principles:

**Short-Term Memory** (`app/memory/short_term.py`)
A sliding window of the last 10 messages. When older turns fall outside the window, a deterministic LLM call (temperature=0) condenses them into a running summary — preventing prompt token bloat without losing factual continuity.

**Semantic Memory** (`app/memory/semantic.py`)
Every assistant turn triggers background extraction of persistent facts about the user: their tech stack, preferences, ongoing projects, identity signals. Facts are stored with 3072-dim Gemini embeddings and upserted idempotently. At query time, cosine similarity search injects the most contextually relevant facts into the prompt.

**Episodic Memory** (`app/memory/episodic.py`)
Completed conversations are distilled into structured episode records — summary, key topics, events, message provenance boundaries. These are embedded and stored in PostgreSQL. Cross-session retrieval lets Ava recall *"what we worked on last Tuesday"* without the user re-explaining context.

**Unified Coordinator** (`app/memory/manager.py`)
`MemoryManager` orchestrates all three layers: context injection before the LLM call, non-blocking background distillation after the turn. Voice TTS latency is never gated on memory writes.

---

### Phase 4 — Multi-Tenancy

Added strict per-user data isolation across every layer:
- `user_id` columns with indexes added to `tasks` and `diary_entries` via safe auto-migration inside `init_db()`
- All CRUD functions in `app/database/controller.py` filter strictly by `user_id`
- A `contextvars.ContextVar` in `app/tools/user_context.py` propagates the active user into tool calls without threading issues
- The agent's `respond()` method synchronizes `user_context` before invoking the tool loop

---

### Phase 5 — Polish: Latency, Bridging & UI

- **Context Bridging:** Detects resume intent (*"Continue where we left off"*, *"What was I discussing?"*) and synthesizes a prioritized continuity prompt from pending tasks + episodic history
- **ChatGPT-style sidebar:** Recent sessions list, `➕ New Chat` button, live memory inspector drawer
- **Dynamic system prompt:** Injects the current year at runtime and enforces `web_search` calls for recency-sensitive queries (model launches, AI news)

---

## Major Challenges & Solutions

### 1. Sarvam STT — Silent Transcription Failure

**Problem:** Sarvam's Realtime WebSocket sends payloads keyed by `"event"` — e.g. `{"event": "transcript.final"}`. Our receiver was reading `"type"`, so every live transcription was silently dropped. The microphone appeared active, but not a word was captured.

**Fix:** Updated `_receiver` in `app/voice/stt.py` to inspect `data.get("event") or data.get("type")` and extract text from `data.get("text") or data.get("transcript")`. Added reconnect pacing to prevent tight loops on socket timeout.

---

### 2. WebRTC Thread Safety & Sample Rate Mismatch

**Problem:** `audio_frame_callback` runs in a background C/WebRTC thread (`async_media_processor_0`). Accessing `st.session_state` from this thread threw `missing ScriptRunContext` exceptions — silently swallowed — so no mic audio reached STT. Compounding this: browsers stream at 48kHz or 44.1kHz; Sarvam expects 16kHz mono PCM16.

**Fix:** Switched to a module-level `_ACTIVE_STT_CLIENT` reference, eliminating all `session_state` access from the audio thread. Integrated `av.AudioResampler(format="s16", layout="mono", rate=16000)` via PyAV to resample every incoming frame to 16kHz mono in-place.

---

### 3. Voice Response Latency

**Problem:** Every turn — including *"Hi"* — triggered full Gemini embedding lookups across both semantic and episodic memory, adding ~200ms before TTS could even begin.

**Fix — three-pronged:**
- **Trivial Turn Bypass:** Greetings and simple math queries skip vector retrieval entirely. Context builds in under 35ms.
- **Concurrent Retrieval:** Semantic and episodic retrievals run in parallel via `ThreadPoolExecutor`, cutting total retrieval latency by ~50%.
- **Non-Blocking Post-Turn:** Fact extraction and eviction summarization are dispatched to a background daemon thread. `record_turn()` returns in under 25ms and never delays TTS.

---

### 4. Multi-Tenancy Inconsistency (Tasks vs. Memory)

**Problem:** Memory tables were correctly scoped per-user from the start, but `tasks` and `diary_entries` had no `user_id` column. Any user could read and modify every other user's data.

**Fix:** Added `user_id` column with an index to both tables via a safe `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migration run inside `init_db()`, followed by a sequence resync to prevent primary key collisions. All controller-level queries now include `WHERE user_id = :active_user`.

---

### 6. LLM Knowledge Cutoff on Recent Events

**Problem:** Asking *"What AI news happened this week?"* or *"What model did ChatGPT just launch?"* returned confident answers that were 1–2 years out of date, sourced entirely from the LLM's frozen training data.

**Fix:** The dynamic system prompt now injects the current year at runtime. Any query containing recency signals — *"latest"*, *"just launched"*, *"recently"*, *"new model"* — **must** invoke the `web_search` tool before answering. Firecrawl fetches live web content to ground the response in current reality.

--
---

## Project Structure

```
Ava/
├── app/
│   ├── agent/
│   │   └── langchain_agent.py      # LangChain resilient agent, dynamic prompting
│   ├── config/
│   │   └── settings.py             # Pydantic settings, env validation
│   ├── database/
│   │   ├── db.py                   # PostgreSQL engine, init_db(), auto-migration
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   └── controller.py           # CRUD with strict user_id scoping
│   ├── memory/
│   │   ├── manager.py              # MemoryManager: orchestration & context bridging
│   │   ├── short_term.py           # Sliding window + eviction summarization
│   │   ├── semantic.py             # Fact extraction + Gemini vector upsert/search
│   │   └── episodic.py             # Episode distillation + pgvector retrieval
│   ├── tools/
│   │   ├── task_tool.py            # Task CRUD tool (multi-tenant)
│   │   ├── diary_tool.py           # Diary CRUD tool (multi-tenant)
│   │   ├── user_context.py         # contextvars thread-safe user scoping
│   │   └── web_search.py           # Firecrawl real-time web search
│   └── voice/
│       ├── stt.py                  # Sarvam AI Realtime WebSocket STT
│       └── tts.py                  # Sarvam AI TTS synthesis
├── scripts/
│   ├── test_memory.py              # Memory subsystem integration tests
│   ├── test_multitenancy.py        # Multi-tenant isolation tests
│   └── test_durability_and_bridging.py  # Latency, durability & bridge tests
├── streamlit_app.py                # Main UI: ChatGPT sidebar + dual voice modes
├── pyproject.toml
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- PostgreSQL 14+ with `pgvector` extension enabled
- API keys: Groq, Google Gemini, Sarvam AI, Firecrawl

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ava.git
cd ava

# Install dependencies
uv sync
```

### Database Setup

```bash
# Enable pgvector extension in PostgreSQL
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Tables are auto-created on first run via init_db()
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/ava_db
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
SARVAM_API_KEY=your_sarvam_key
FIRECRAWL_API_KEY=your_firecrawl_key
```

### Run

```bash
uv run streamlit run streamlit_app.py
```

---

## Running Tests

```bash
# Memory subsystem (short-term, semantic, episodic)
uv run python scripts/test_memory.py

# Multi-tenant isolation (tasks, diary, agent flow)
uv run python scripts/test_multitenancy.py

# Latency benchmarks, DB durability, context bridging
uv run python scripts/test_durability_and_bridging.py
```

**Verified benchmarks:**

| Metric | Result |
|---|---|
| Trivial turn context build (greeting bypass) | `< 35ms` ✅ |
| `record_turn()` post-processing return time | `< 25ms` ✅ |
| Multi-tenant task isolation (Alice vs Bob) | Verified ✅ |
| Cross-session episodic recall | Verified ✅ |
| PostgreSQL-only engine dialect | Verified ✅ |

---

<div align="center">

*Built with care — from a simple voice loop to a cognitive AI system*

</div>
