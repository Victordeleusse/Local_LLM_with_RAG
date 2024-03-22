# Local_LLM

Running local LLMs with Ollama to perform RAG for answering questions based on sample PDFs

## Requirements

Ollama - Please download the latest version **https://ollama.com/download**

As the project is running through a Dockerized environment, setting up a virtual environment is not necessary. Please ensure that ports **1024** and **8000** are open.

## Running the Project

Note: The first time you run the project, it will download the necessary models from Ollama for the LLM and embeddings. This is a one-time setup process and may take some time depending on your internet connection.

The specificity of this project lies in its Dockerized environment. In order to make the various API calls required for this project, the use of a reverse-proxy (Nginx) was necessary.

1. Please run ``` docker-compose build ``` and then run the app docker ```docker-compose run -i app``` to launch the application and chat with the model.
By default, **mistral** llm model is used, with **nomic-embed-text** embedding-model. Please drop your files into Files folder

2. To specify specific model / embedding-model / path for your files, please run ``` docker-compose run -i app python app.py -m MODEL -e EMBEDDING_MODEL -p YOUR_PATH``` 

This will load the PDFs and Markdown files, generate embeddings, query the collection, and answer the question defined in app.py keeping up a conversation history to ensure a proper conversation.

## Technologies Used

- [Langchain](https://python.langchain.com/docs/get_started/introduction): A Python library for working with Large Language Model.
- [Ollama](https://github.com/ollama/ollama): A platform for running Large Language models locally.
- [Chroma](https://docs.trychroma.com/): A vector database for storing and retrieving embeddings.
- [PyPDF](https://github.com/py-pdf/pypdf): A Python library for reading and manipulating PDF files.
