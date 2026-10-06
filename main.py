"""Command-line chat app: LangChain + in-memory RAG + dynamic model selection."""

import sys
import uuid

from chain import clear_history, respond
from rag import DATA_DIR, ingest

HELP = """\
Commands:
  /help            show this help
  /models          list available chat models
  /model <name>    switch to a model (e.g. /model deepseek-r1:7b)
  /reload          re-embed documents from data/
  /clear           clear conversation history
  /quit            exit
Anything else is sent to the model."""


def list_models() -> list[str]:
    import ollama

    models = ollama.list().models
    return [m.model for m in models if "embed" not in m.model.lower()]


def choose_model(models: list[str]) -> str:
    print("Available models:")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input(f"Select model [1-{len(models)}] (default 1): ").strip()
        if not choice:
            return models[0]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        if choice in models:
            return choice
        print("Invalid selection, try again.")


def main() -> None:
    try:
        models = list_models()
    except Exception as exc:
        print(f"Cannot reach Ollama ({exc}). Is `ollama serve` running?", file=sys.stderr)
        sys.exit(1)
    if not models:
        print("No chat models found. Pull one with `ollama pull <model>`.", file=sys.stderr)
        sys.exit(1)

    model_name = choose_model(models)
    chunks = ingest()
    print(f"\nIndexed {chunks} chunks from {DATA_DIR}/ into in-memory vector store")
    print(f"Model: {model_name}")
    print(f"Type /help for commands.\n")

    session_id = uuid.uuid4().hex

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user_input:
            continue

        cmd, _, arg = user_input.partition(" ")
        if cmd in ("/quit", "/exit"):
            print("bye")
            break
        if cmd == "/help":
            print(HELP)
            continue
        if cmd == "/models":
            for name in list_models():
                marker = "*" if name == model_name else " "
                print(f" {marker} {name}")
            continue
        if cmd == "/model":
            available = list_models()
            if arg not in available:
                print(f"Unknown model '{arg}'. Available: {', '.join(available)}")
            else:
                model_name = arg
                print(f"Switched to {model_name}")
            continue
        if cmd == "/reload":
            chunks = ingest()
            print(f"Re-indexed {chunks} chunks from {DATA_DIR}/")
            continue
        if cmd == "/clear":
            clear_history(session_id)
            print("Conversation history cleared.")
            continue

        try:
            reply, used_rag = respond(user_input, model_name, session_id)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue

        if used_rag:
            reply += "\n\n(inferred from retrieved documents)"
        print(f"{model_name}> {reply}")


if __name__ == "__main__":
    main()
