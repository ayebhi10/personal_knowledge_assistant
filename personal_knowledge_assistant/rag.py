import logging
import os
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

    if not os.path.exists(db_dir):
        documents = load_documents()

        if not documents:
            raise RuntimeError(f"No documents found in {DOCS_DIR}. Add files before building the vectorstore.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(documents)
        split_docs = [doc for doc in split_docs if doc.page_content.strip()]

        logging.info(f"Building vectorstore with {len(split_docs)} non-empty chunks, in batches...")

        # Create the (empty) vectorstore first, then add documents in batches
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embedding_instance)

        batch_size = 50  # tune this down further if it still crashes
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            vectorstore.add_documents(batch)
            logging.info(f"Added batch {i // batch_size + 1} / {(len(split_docs) - 1) // batch_size + 1}")
    else:
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embedding_instance)

    return vectorstore
