from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document


def load_documents(path):
    """
    Loads documents from the specified directory path, supporting load from PDF, Markdown, and HTML 
    documents by utilizing different loaders for each file type. 
    Args:
        path (str): The path to the directory containing documents to load.
    Returns:
        List[Document]: A list of loaded documents.
    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    # # Initialize the loaders
    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }
    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    # Return a list of loaded documents    
    return docs