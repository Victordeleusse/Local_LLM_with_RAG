from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from models import check_if_model_is_available
from load_docs import load_documents
import argparse
import sys

import ollama
from ollama import Client

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
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    )
    return db4

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
    query = "What is the date of the start of the battle ?"
    docs = vector_store.similarity_search(query)
    # print(type(docs)) # <class 'list'>
    # # docs_dict = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    print(docs[0].page_content)
    # llm = Ollama(
    #     model=llm_model_name,
    #     callbacks=[StreamingStdOutCallbackHandler()],
    # )
    # my_retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 3, "score_threshold": 0.5},
    #     )
    
    # llm = Ollama(model=llm_model_name, base_url="http://localhost:11434")
    llm = Ollama(model=llm_model_name, base_url="http://ollama:11434")
    response = llm.generate(model=llm_model_name, prompts=["Say hello in JSON"])
    print(response)
    
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=my_retriever,
    #     chain_type_kwargs={"prompt": PROMPT},
    # )
    
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=my_retriever, return_source_documents=True) 
    # response = qa(query)
    
    
    
    
    # response = ollama.generate(model=llm, prompt=PROMPT, context=docs, stream=True)
    # print(response)
    # response = llm.generate(prompt="Say hello for test in JSON object. Object:", stream=False)
    # print(response['response'])
    
    # prompt = "What is the difference between an adverb and an adjective?" 
    # chain = ({"context":  my_retriever | format_docs, "question": RunnablePassthrough()}
    #                   | PROMPT
    #                   | llm
    #                   | StrOutputParser())
    
    # response = chain.invoke(query)
    # print(response)
    
    # json_data = []
    # for chunk in response:
    #     # Assuming each chunk is a dictionary
    #     json_data.append(chunk)

    
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
    