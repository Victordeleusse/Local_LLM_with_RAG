from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
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

def load_documents_into_database(llm_model_name, model_name, documents_path):
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
    
    # user_input = "What is the date of the start of the battle"
    # docs = db4.similarity_search(user_input)
    # print(docs[0].page_content)
    
    retriever = db4.as_retriever()
    # Prompt
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Local LLM
    ollama_llm = llm_model_name
    model_local = ChatOllama(model=ollama_llm)

    # Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model_local
        | StrOutputParser()
    )
    # Question
    chain.invoke("What is the date of the start of the battle?")
    
    return db4, retriever
    # return db4

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
        db, retriever = load_documents_into_database(llm_model_name, embedding_model_name, documents_path)

    except FileNotFoundError as e:
        print(e)
        sys.exit()

    # llm = Ollama(model=llm_model_name,callbacks=[StreamingStdOutCallbackHandler()],)

    # # Prompt
    # template = """Answer the question based only on the following context:
    # {context}

    # Question: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)

    # # Local LLM
    # ollama_llm = llm_model_name
    # model_local = ChatOllama(model=ollama_llm)

    # # Chain
    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | model_local
    #     | StrOutputParser()
    # )
    # # Question
    # chain.invoke("What is the date of the start of the battle?")

    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=db.as_retriever(search_kwargs={"k": 8}),
    #     chain_type_kwargs={"prompt": PROMPT},
    # )

    # while True:
        # try:
    # print("In progress ....")
# try:
#     user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
#     if user_input.lower() == "exit":
#         break
    # user_input = "What is the date of the start of the battle"
    # docs = db.similarity_search(user_input)
    # print(docs[0].page_content)
        #     qa_chain.invoke({"query": user_input})
        # except KeyboardInterrupt:
            # break
        
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
    