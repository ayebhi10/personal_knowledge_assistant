import logging
from typing import Any
from langchain.agents.agent import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

from config import SYSTEM_PROMPT, RETRIEVAL_K
from model import LLMModel
from providers import get_llm


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def before_model(self, state: State) -> dict[str, Any] | None:
        msg = state["messages"][-1]
        query = str(msg.content) if msg else ""

        docs = self.vector_store.similarity_search(query, k=RETRIEVAL_K)
        logging.info(f"Retrieved {len(docs)} documents for query: {query}")

        context = "\n\n".join(f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs)

        system_message = SystemMessage(content=SYSTEM_PROMPT + "\n\n" + context)

        return {
            "messages": [system_message],
            "context": docs,
        }
    

def build_agent(llm_model: LLMModel, vector_store) -> Any:
    llm = get_llm(llm_model)
    agent = create_agent(
        llm=llm,
        middleware=[RetrieveDocumentsMiddleware(vector_store)],
        state_cls=State,
    )
    return agent