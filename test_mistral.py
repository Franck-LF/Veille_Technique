
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

mistra_api_key = os.getenv('MISTRAL_API_KEY')

# Initialisation du client
client = Mistral(api_key=mistra_api_key)

# Texte à encoder
texts = ["Bonjour, ceci est un exemple de texte", "Un deuxième texte à encoder"]

# Appel à l'API d'embedding
response = client.embeddings.create(
    model="mistral-embed",  # Modèle d'embedding
    inputs=texts
)

# Récupération des embeddings
embeddings = [embedding.embedding for embedding in response.data]
print(f"Nombre d'embeddings : {len(embeddings)}")
print(f"Dimension d'un embedding : {len(embeddings[0])}")

