

from sentence_transformers import SentenceTransformer
from data import SENTENCES
from benchmark_utils import benchmark

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_local(texts):
    return model.encode(texts).tolist()

print(benchmark(embed_local, SENTENCES))


from sentence_transformers import SentenceTransformer

# Charger un modèle (gratuit, s'exécute localement)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Encoder des textes
texts = ["Bonjour, ceci est un texte", "Hello, this is a text"]
embeddings = model.encode(texts)

print(f"Shape: {embeddings.shape}")  # Ex: (2, 384)
