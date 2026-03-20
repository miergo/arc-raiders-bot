# ARC-RAIDERS WikiBotttttttt

A wiki-style chatbot for ARC Raiders powered by a local Ollama LLM, DuckDuckGo web search, and a ChromaDB RAG cache. Python/FastAPI backend + React frontend.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Ollama** running locally with two models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ollama serve   # if not already running
  ```

## Quick Start

```bash
# 1. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults work for local Ollama
python server.py              # starts API on http://localhost:8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                   # starts UI on http://localhost:5173
```

Open http://localhost:5173 and start asking about ARC Raiders.

## Configuration

All config lives in `backend/.env` (copy from `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for RAG cache |
| `CHROMA_DATA_DIR` | `data/chroma` | ChromaDB storage path |
| `CACHE_SIMILARITY_THRESHOLD` | `0.75` | Min similarity to use cached answers |
| `VERBOSE_TOOL_CALLS` | `0` | Set to `1` for debug output |

## Project Structure

```
backend/
  server.py                    FastAPI entry point
  arc_raider_bot/
    config.py                  Env var loading
    prompts.py                 System prompts
    models.py                  Pydantic models (agent + API)
    agent.py                   LLM agents + pipeline
    sessions.py                Conversation history store
    knowledge_cache.py         ChromaDB RAG cache
    cli.py                     Interactive terminal client
frontend/
  src/
    api.ts                     API client
    App.tsx                    Chat UI shell
    components/                ChatWindow, ChatInput, BotMessage
```

## Other Ways to Run

```bash
cd backend

# Interactive CLI (no frontend needed)
python -m arc_raider_bot.cli

# Smoke test with verbose output
VERBOSE_TOOL_CALLS=1 python run_demo.py
```
