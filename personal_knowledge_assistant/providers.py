# providers.py
from enum import Enum
from model import EmbeddingModel, LLMModel

class Provider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

EMBEDDING_PROVIDER: dict[EmbeddingModel, Provider] = {
    EmbeddingModel.OLLAMA_NOMIC: Provider.OLLAMA,
    EmbeddingModel.OLLAMA_MXBAI: Provider.OLLAMA,
    EmbeddingModel.OPENAI_3_SMALL: Provider.OPENAI,
}

LLM_PROVIDER: dict[LLMModel, Provider] = {
    LLMModel.OLLAMA_LLAMA3: Provider.OLLAMA,
    LLMModel.OLLAMA_QWEN: Provider.OLLAMA,
    LLMModel.GPT4O: Provider.OPENAI,
}