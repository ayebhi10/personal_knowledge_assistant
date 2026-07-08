import logging
from model import EmbeddingModel, LLMModel
from rag import get_vectorstore
from agent import build_agent

def run_agent(embedding_model: EmbeddingModel, llm_model: LLMModel):
    vector_store = get_vectorstore(embedding_model)
    agent = build_agent(llm_model, vector_store)

    logging.info(f"Agent initialized with LLM: {llm_model.value} and Embedding Model: {embedding_model.value}")
    
    while True:
        question = input("You: ").strip()
        if not question or question.lower() == "exit":
            break

        result = agent.inoke({
            "messages": [{"role": "user", "content": question}],
            "context": [],
        })

        logging.info(f"Agent response: {result}")
        logging.info("Sources : ")

        seen = set()
        for doc in result.get("context", []):
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                logging.info(f"- {source}")
                seen.add(source)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_agent(EmbeddingModel.OLLAMA_NOMIC, LLMModel.OLLAMA_LLAMA3)