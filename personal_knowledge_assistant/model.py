from enum import Enum

class EmbeddingModel(str, Enum):
    OLLAMA_NOMIC = "nomic-embed-text"
    QWEN3_EMBEDD = "qwen3-embedding"
    OLLAMA_MXBAI = "mxbai-embed-large"
    OPENAI_3_SMALL = "text-embedding-3-small"


class LLMModel(str, Enum):
    OLLAMA_QWEN   = "qwen-3.5:4b"
    OLLAMA_GEMMA3 = "gemma3:4b"
    OLLAMA_GEMMA4 = "gemma4:e4b"
    OLLAMA_LLAMA3 = "llama3.1"
    GPT4O         = "gpt-4o"
