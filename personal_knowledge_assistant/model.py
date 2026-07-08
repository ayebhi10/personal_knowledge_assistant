from enum import Enum

class EmbeddingModel(str, Enum):
    OLLAMA_NOMIC = "nomic-embed-text"
    OLLAMA_MXBAI = "mxbai-embed-large"
    OPENAI_3_SMALL = "text-embedding-3-small"


class LLMModel(str, Enum):
    OLLAMA_QWEN   = "qwen-3.5:4b"
    OLLAMA_GEMMA3 = "gemma3:4b"
    OLLAMA_LLAMA3 = "llama3.1"
    GPT4O         = "gpt-4o"
