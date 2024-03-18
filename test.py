import ollama

if __name__ == "__main__":
    print("Launching test")
    response = ollama.pull("llama2")
    
    print(response)
    