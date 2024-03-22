from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from models import check_if_model_is_available
from load_docs import load_documents
import argparse
import sys

from hist import getChatChain

import warnings
warnings.filterwarnings("ignore")

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_documents_into_database(model_name, documents_path):
    """
    Loads documents from the specified directory into the Chroma database after splitting the text into chunks.
    Returns: Chroma, database with loaded documents.
    """
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    db = chromadb.HttpClient(host="chroma", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

    oembed = OllamaEmbeddings(base_url="http://ollama:11434", model=model_name)
    
    vectorstore = Chroma.from_documents(client=db, documents=documents, embedding=oembed)
    
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def global_execution_process(llm_model_name, embedding_model_name, documents_path):
    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()
    # Creating database form documents
    try:
        print(f"Loading documents from : {documents_path}")
        vector_store = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()
    
    llm = Ollama(model=llm_model_name, base_url="http://ollama:11434")
    
    qa_chain = getChatChain(llm, vector_store)

    while True:
        try:
            user_input = input('\n\nPlease enter your question (or type "exit" to end): ')
            if user_input.lower() == "exit":
                break
            response = qa_chain(user_input)
            print(response)
        except KeyboardInterrupt:
            break

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Local_LLM',
                    description='Run local LLM with RAG with Ollama.')
    parser.add_argument('-m', '--model', default="mistral",
        help="The name of the LLM model to use.")
    parser.add_argument('-e', '--embedding_model', default="nomic-embed-text",
        help="The name of the embedding model to use.")
    parser.add_argument('-p', '--path', default="./Files",
        help="The path to the directory containing documents to analyse.")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_arguments()
    global_execution_process(args.model, args.embedding_model, args.path)
    