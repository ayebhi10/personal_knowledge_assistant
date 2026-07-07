from functools import lru_cache
from model import EmbeddingModel
from registry import Registry

embedding_registry: Registry[EmbeddingModel, object] = Registry("embedding_registry")

@embedding_registry.register(EmbeddingModel.OLLAMA_NOMIC)
@embedding_registry.register(EmbeddingModel.OLLAMA_MXBAI)
def _load_ollama_embedding(model: EmbeddingModel, *args, **kwargs) -> object:
    from langchain_ollama import OllamaEmbedding
    return OllamaEmbedding(model=model, *args, **kwargs)


@embedding_registry.register(EmbeddingModel.OPENAI_3_SMALL)
def _load_openai_embedding(model: EmbeddingModel, *args, **kwargs) -> object:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model, *args, **kwargs)


@embedding_registry.default
def _load_huggingface_embedding(model: EmbeddingModel, *args, **kwargs) -> object:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model=model, *args, **kwargs)


@lru_cache(maxsize=None)
def get_embedding_model(model:EmbeddingModel, *args, **kwargs) -> object:
    """Get an embedding model instance from the registry."""
    return embedding_registry.create(model, model)