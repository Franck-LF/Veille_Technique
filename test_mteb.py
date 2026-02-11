from sentence_transformers import SentenceTransformer
from data import SENTENCES
from benchmark_utils import benchmark
import mteb

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_local(texts):
    return model.encode(texts).tolist()


# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

task = mteb.get_tasks(["STSBenchmark"])
evaluator = mteb.MTEB(task)

model = mteb.get_model("BBAI/bge-base-en")
evaluator.run(model)