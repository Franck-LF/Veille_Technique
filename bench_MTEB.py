
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
evaluation = MTEB(tasks=["STSBenchmark"])
evaluation.run(model)
