

from sentence_transformers import SentenceTransformer
from data import SENTENCES
from benchmark_utils import benchmark

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_local(texts):
    return model.encode(texts).tolist()

print(benchmark(embed_local, SENTENCES))
