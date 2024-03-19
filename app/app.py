# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import check_if_model_is_available
from load_docs import load_documents
import argparse
import sys
import ollama
import uuid



TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

PROMPT_TEMPLATE = """
### Instruction:
You're helpful assistant, who answers questions based upon provided research in a distinct and clear way.

## Research:
{context}

## Question:
{question}
"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

def load_documents_into_database(model_name, documents_path):
    """
    Loads documents from the specified directory into the Chroma database after splitting the text into chunks.
    Returns: Chroma, database with loaded documents.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    db = chromadb.HttpClient(host="chroma", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
    db.reset()
    collection = db.create_collection("my_collection")
    # collection = db.get_or_create_collection(name="my_collection")
    for doc in documents:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )
    db4 = Chroma(
        client=db,
        collection_name="my_collection",
        # embedding_function=OllamaEmbeddings(model=model_name),
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    )
    
    # return db4, retriever
    return db4

def global_execution_process(llm_model_name, embedding_model_name, documents_path):
    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(llm_model_name)
        print(f"MODEL CHECKED : {llm_model_name}")
        check_if_model_is_available(embedding_model_name)
        print(f"EMBEDDING CHECKED : {embedding_model_name}")
    except Exception as e:
        print(e)
        sys.exit()
        
    # Creating database form documents
    try:
        print(f"Loading documents from : {documents_path}")
        # db, retriever = load_documents_into_database(llm_model_name, embedding_model_name, documents_path)
        db = load_documents_into_database(embedding_model_name, documents_path)

    except FileNotFoundError as e:
        print(e)
        sys.exit()

    query = "What is the date of the start of the battle ?"
    docs = db.similarity_search(query)
    print(docs[0].page_content)
    
    llm = Ollama(
        model=llm_model_name,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    my_retriever = db.as_retriever(search_kwargs={"k": 8})
    
    # response = ollama.generate(model='mistral', prompt='Why is the sky blue?')
    response = ollama.list()
    print(response)
    # print(response)
    # # Build prompt
    # template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    # {context}
    # Question: {question}
    # Helpful Answer:"""
    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=my_retriever,
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    # )
    
    # result = qa_chain({"query": query})
    # print(result)
    # result["result"]
    
    
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
    print("Launching Test")
    args = parse_arguments()
    print(f"MODEL set up : {args.model}")
    global_execution_process(args.model, args.embedding_model, args.path)
    
    # response = ollama.list()
    # response = ollama.generate(model='mistral', prompt='Why is the sky blue?')
    # print(response)
    