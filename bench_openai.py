
import os
from openai import OpenAI
from data import SENTENCES
from benchmark_utils import benchmark
from dotenv import load_dotenv


load_dotenv()

key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=key)

def embed_openai(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [x.embedding for x in resp.data]

print(benchmark(embed_openai, SENTENCES))
