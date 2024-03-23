FROM python:3.9-slim

RUN apt-get update && apt-get install -y curl && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install -U langchain
# RUN pip install langchain-nomic 
# RUN pip install langchain-core

CMD ["python", "app.py"]