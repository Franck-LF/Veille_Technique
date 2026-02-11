
import os
import cohere
from data import SENTENCES
from benchmark_utils import benchmark
from dotenv import load_dotenv
import numpy as np

load_dotenv()

cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialisation avec votre clé API
co = cohere.Client(cohere_api_key)

# Liste de textes à encoder
texts = ["Bonjour, ceci est un texte en français", 
         "Hello, this is an English text",
         "Hola, este es un texto en español"]

# Appel à l'API d'embedding
response = co.embed(
    texts=texts,
    model="embed-multilingual-v3.0",  # Modèle multilingue
    input_type="search_document"      # Type d'utilisation
)

# Récupération des embeddings
embeddings = response.embeddings

# Affichage des résultats
print(f"Nombre de textes : {len(embeddings)}")
print(f"Dimension d'un embedding : {len(embeddings[0])}")
print(f"Premier embedding (10 premières valeurs) : {embeddings[0][:10]}")

# Conversion en numpy array pour manipulation
embeddings_array = np.array(embeddings)
print(f"\nShape du tableau d'embeddings : {embeddings_array.shape}")
