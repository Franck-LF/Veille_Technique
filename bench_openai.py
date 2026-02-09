
from openai import OpenAI
from data import SENTENCES
from benchmark_utils import benchmark

client = OpenAI(api_key="YOUR_KEY")

def embed_openai(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [x.embedding for x in resp.data]

print(benchmark(embed_openai, SENTENCES))
