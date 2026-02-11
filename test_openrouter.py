
import os
import requests
import json
from dotenv import load_dotenv


load_dotenv()

openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

response = requests.post(
  url="https://openrouter.ai/api/v1/embeddings",
  headers={
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "thenlper/gte-base",
    "input": "Your text string goes here",
    "encoding_format": "float"
  })
)

print(f"Longueur de l'emmbedding: {len(response.json()['data'][0]['embedding'])}")


print(response.json()['data'][0]['embedding'][:10])
