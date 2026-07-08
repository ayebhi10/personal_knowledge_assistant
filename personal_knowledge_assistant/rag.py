from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader

from config import DOCS_DIR, DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K, db_dir_for
from model import EmbeddingModel
from providers import get_embedding_model, get_llm

def load_documents():
    """Load documents from the DOCS_DIR."""
    documents = []
    for file_path in Path(DOCS_DIR).glob("**/*"):
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            documents.append(Document(page_content=text, metadata={"source": str(file_path)}))
        elif file_path.suffix.lower() in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": str(file_path)}))
    return documents


def get_vectorstore(embedding_model: EmbeddingModel):
    """Get or create a Chroma vectorstore."""
    embedding_instance = get_embedding_model(embedding_model)
    db_dir = db_dir_for(embedding_model)
    if not db_dir.exists():
        documents = load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(split_docs, embedding=embedding_instance, persist_directory=DB_DIR)
        vectorstore.persist()
    else:
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_instance)
    return vectorstore
