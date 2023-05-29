import requests

url = "http://191.95.165.174:5050/model-llm"
data = {
    'prompt': 'find me a job as a data engineer'
}

response = requests.post(url, json=data)

print(response.json())
