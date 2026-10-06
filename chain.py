"""Pure LangChain chat: LCEL chain with RAG vs plain prompt, in-memory history."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from rag import retrieve

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using the context below \
when it is relevant. If the context does not contain the answer, use your own \
knowledge and say when you are doing so.

Context:
{context}
"""

CHAT_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question concisely."

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CHAT_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

_histories: dict[str, list[BaseMessage]] = {}


def get_history(session_id: str) -> list[BaseMessage]:
    return _histories.setdefault(session_id, [])


def clear_history(session_id: str) -> None:
    get_history(session_id).clear()


def respond(query: str, model_name: str, session_id: str) -> tuple[str, bool]:
    """Run one chat turn through the chain. Returns (answer, used_rag)."""
    docs = retrieve(query)
    history = get_history(session_id)

    prompt = RAG_PROMPT if docs else CHAT_PROMPT
    inputs: dict = {"input": query, "history": list(history)}
    if docs:
        inputs["context"] = "\n\n---\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        )

    chain = prompt | ChatOllama(model=model_name) | StrOutputParser()
    answer = chain.invoke(inputs)

    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=answer))
    return answer, bool(docs)
