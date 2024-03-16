import ollama
from langchain_community.llms import Ollama
from tqdm import tqdm
import subprocess
import json

# def __is_model_available_locally(model_name):
#     try:
#         print(f"Retrieving model: {model_name}")
#         curl_cmd = f"curl -X POST http://localhost:11434/api/show -H 'Content-Type: application/json' -d '{{\"name\": \"{model_name}\"}}'"
#         result = subprocess.run(curl_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         response = result.stdout
#         error = result.stderr
#         if error:
#             print(f"Error retrieving model: {error}")
#             return False
#         response_data = json.loads(response)
#         print(response_data)
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
#     except Exception as e:
#         print(f"Model not found or an error occurred: {e}")
#     return False

def __is_model_available_locally(model_name):
    try:
        print(f"Retrieving model: {model_name}")
        model = ollama.pull(model_name)
        # model.run()
        print(f"Model: {model_name} running")
        return True
    except Exception as e:
        print(f"Model not found or an error occurred: {e}")
        return False

def check_if_model_is_available(model_name):
    """
    Ensures that the specified model is available locally.
    If the model is not available, it attempts to pull it from the Ollama repository.
    Args:
        model_name (str): The name of the model to check.
    Raises:
        ollama.ResponseError: If there is an issue with pulling the model from the repository.
    """
    if not __is_model_available_locally(model_name):
        try:
            for progress in ollama.pull(model_name, stream=True):
                digest = progress.get("digest", "")
                if not digest:
                    print(progress.get("status"))
        except:
            raise Exception(f"Unable to find model '{model_name}', please check the name and try again.")