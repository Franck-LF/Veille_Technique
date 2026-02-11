
import os
from mistralai import Mistral
from data import SENTENCES
from benchmark_utils import benchmark
from dotenv import load_dotenv

load_dotenv()

mistra_api_key = os.getenv('MISTRAL_API_KEY')

# Initialisation du client
client = Mistral(api_key=mistra_api_key)

def embed_mistral(texts):
    response = client.embeddings.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        # model="mistral-embed",
        inputs=texts
    )
    return [embedding.embedding for embedding in response.data]

print(benchmark(embed_mistral, SENTENCES))
