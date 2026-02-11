
import time
import numpy as np

def benchmark(fn, sentences):
    print("======= START =======")
    start = time.perf_counter()
    vectors = fn(sentences)
    # print(vectors)
    end = time.perf_counter()
    print("=======  END  =======")

    duration = end - start
    n = len(sentences)

    return {
        "count": n,
        "duration_sec": duration,
        "latency_avg_ms": (duration / n) * 1000,
        "throughput": n / duration,
        "dim": len(vectors[0]),
    }

