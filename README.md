# langchain-demo

CLI chat application built with pure LangChain (LCEL) featuring:

- **In-memory RAG** — documents in `data/` are chunked and embedded into LangChain's `InMemoryVectorStore` (no disk persistence; re-indexed each run)
- **Dynamic model selection** — pick any local Ollama chat model at startup, switch anytime with `/model`
- **Context retention** — conversation history kept per session in memory; `/clear` empties it
- **Open models** — chat via Ollama (`qwen2.5:3b`, `deepseek-r1:7b`), embeddings via `nomic-embed-text`

## Prerequisites

- [Ollama](https://ollama.com) running locally
- Models:

```bash
ollama pull qwen2.5:3b
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

## Run

```bash
uv run main.py
```

## Commands

| Command | Description |
|---|---|
| `/help` | show help |
| `/models` | list chat models (`*` marks active) |
| `/model <name>` | switch model, e.g. `/model deepseek-r1:7b` |
| `/reload` | re-embed documents from `data/` |
| `/clear` | clear conversation history |
| `/quit` | exit |

## Local models (Ollama)

All inference runs locally through [Ollama](https://ollama.com) — no API keys,
no cloud calls. Once the `ollama pull` steps above are done, the demo works
fully offline.

| Role | Model | Used in | Why |
|---|---|---|---|
| Chat (default) | `qwen2.5:3b` | `chain.py` via `ChatOllama` | small and fast for interactive replies |
| Chat (reasoning) | `deepseek-r1:7b` | `chain.py` via `ChatOllama` | slower, but reasons through complex questions |
| Embeddings | `nomic-embed-text` | `rag.py` via `OllamaEmbeddings` | open embedding model for the vector store |

### How the local models are wired in

- **Embedding (RAG side)** — at startup `ingest()` chunks the files in `data/`
  and embeds each chunk with `nomic-embed-text` into LangChain's
  `InMemoryVectorStore`. On every turn, the user's message is embedded with the
  *same* model to find the top matching chunks (cosine score ≥ 0.5). Ingestion
  and query embedding both hit the local Ollama server.
- **Chat (generation side)** — every turn builds `ChatOllama(model=<selected>)`
  from the currently selected model name. The selection is just a variable, so
  `/model` swaps the model for the next turn while the conversation history
  stays intact — the new model picks up the same context.
- **Model discovery** — at startup the CLI calls `ollama.list()` and offers
  every local chat model (names containing `embed` are filtered out).
  `/models` and `/model` re-run the same listing, so models you pull while the
  app is open appear without a restart.

### Using other local models

```bash
ollama pull llama3.2:3b    # then pick it at startup or via /model
ollama pull mistral:7b
```

Any instruction-tuned Ollama model works as the chat model. The embedding
model is fixed to `nomic-embed-text` in `rag.py` (`EMBED_MODEL`) — if you
change it, edit that constant and restart the app so the store is re-embedded
consistently.

## Layout

```
main.py   CLI loop, commands, model selection
chain.py  LCEL chain (prompt | llm | parser), RAG vs plain prompt, chat history
rag.py    in-memory vector store + ingestion from data/
data/     sample documents indexed for RAG
```

## How it works

Each turn calls `respond()` in `chain.py`: the query is searched against the
in-memory vector store (cosine score threshold 0.5). Relevant chunks select the
RAG prompt (context injected into the system message); otherwise the plain chat
prompt is used. The chosen prompt is piped with `ChatOllama` and a
`StrOutputParser` into one LCEL chain. Message history is kept per session in
memory and fed back into the prompt on every call, so switching models
mid-conversation (`/model`) keeps the full context.

> Note: the `langgraph` package is present in the environment only because
> `langchain` 1.x depends on it transitively — no graph code is used here.
