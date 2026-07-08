# providers.py
from functools import lru_cache
from model import EmbeddingModel, LLMModel
from registry import Registry

embedding_registry: Registry[EmbeddingModel, object] = Registry("embedding model")
llm_registry: Registry[LLMModel, object] = Registry("LLM")

@embedding_registry.register(EmbeddingModel.OLLAMA_NOMIC)
@embedding_registry.register(EmbeddingModel.OLLAMA_MXBAI)
def _load_ollama_embedding(model: EmbeddingModel):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model.value)

@llm_registry.register(LLMModel.OLLAMA_LLAMA3)
@llm_registry.register(LLMModel.OLLAMA_QWEN)
@llm_registry.register(LLMModel.OLLAMA_GEMMA3)
def _load_ollama_llm(model: LLMModel):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model.value, temperature=0)


@lru_cache(maxsize=None)
def get_embedding_model(model: EmbeddingModel):
    return embedding_registry.create(model, model)

@lru_cache(maxsize=None)
def get_llm(model: LLMModel):
    return llm_registry.create(model, model)