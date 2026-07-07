from functools import lru_cache
from model import LLMModel
from registry import Registry

llm_registry: Registry[LLMModel, object] = Registry("llm_registry")


@llm_registry.register(LLMModel.OLLAMA_QWEN)
@llm_registry.register(LLMModel.OLLAMA_LLAMA3)
def _load_ollama_llm(model: LLMModel, *args, **kwargs) -> object:
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, *args, **kwargs)


@llm_registry.register(LLMModel.GPT4O)
def _load_openai_llm(model: LLMModel, *args, **kwargs) -> object:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, *args, **kwargs)


@llm_registry.default
def _loading_generic_llm(model: LLMModel):
    """Default factory for LLMs not explicitly registered."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model)