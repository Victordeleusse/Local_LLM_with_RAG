import requests

def send_curl_request():
    url = "http://localhost/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama2",
        "prompt": "Why is the sky blue?"
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

if __name__ == "__main__":
    print("Launching test")
    response = send_curl_request()
    print(response)