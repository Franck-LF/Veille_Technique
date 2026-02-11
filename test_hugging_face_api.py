
import requests

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer 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"}

def get_embeddings(texts):
    """Récupère les embeddings via l'API"""
    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    return response.json()

# Utilisation
texts = ["Bonjour, ceci est un texte", "Hello, this is a text"]
embeddings = get_embeddings(texts)
print(f"Nombre d'embeddings : {len(embeddings)}")
print(embeddings[0])
# print(f"Dimensions : {len(embeddings[0])}")
