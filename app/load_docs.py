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

    # Initialize the loaders
    markdown_loader = TextLoader()
    pdf_loader = PyPDFLoader()

    # Initialize the directory loader
    directory_loader = DirectoryLoader()

    docs = []
    # Load all files from the directory
    for file in directory_loader.load(path):
        # Check the file extension and use the appropriate loader
        if file.endswith('.md'):
            document = markdown_loader.load(file)
        elif file.endswith('.pdf'):
            document = pdf_loader.load(file)
        docs.extend(document)
    # Return a list of loaded documents    
    return docs