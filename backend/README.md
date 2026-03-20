# ARC-RAIDERS WikiBot — Backend

Python-only LLM agent that answers ARC Raiders questions using a local Ollama model and DuckDuckGo web search.

## Prerequisites

- **Python 3.10+**
- **Ollama** running locally with the `llama3.1:8b` model pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama serve          # if not already running
  ```

## Setup

```bash
cd backend

# Create a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Copy and edit env vars (defaults work for local Ollama)
cp .env.example .env
```

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name as shown by `ollama list` |
| `VERBOSE_TOOL_CALLS` | `0` | Set to `1` to print tool call debug info |

## Run

### Quick smoke test

```bash
python run_demo.py
```

Runs one hardcoded question with verbose tool output enabled, then prints the structured `ArcRaidersResponse` (answer + sources).

### Interactive CLI

```bash
python cli.py 
```

Type ARC Raiders questions and get wiki-style answers. Type `quit` to exit.


