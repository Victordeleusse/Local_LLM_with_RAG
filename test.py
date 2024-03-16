import ollama

if __name__ == "__main__":
    # model = ollama.pull("mistral")
    response = ollama.pull("llama2")
    
    print(response)
    