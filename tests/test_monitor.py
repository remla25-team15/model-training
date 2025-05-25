'''
    Monitoring
'''
import numpy as np
from memory_profiler import memory_usage
import time

SAMPLE_INPUT = np.random.rand(1, 1421)

# Constants for performance limits based on my run - Adjust these later
MAX_MEMORY_MB = 500
MAX_LATENCY_MS = 1
MIN_THROUGHPUT = 1000

def test_prediction_memory(trained_model):
    """
    Test memory usage during prediction
    """

    def predict():
        trained_model.predict(SAMPLE_INPUT)

    # Measure memory usage
    peak_mem = max(memory_usage(predict, interval=0.1, timeout=1))
    print(f"Peak memory usage: {peak_mem:.1f}MB")

    assert peak_mem <= MAX_MEMORY_MB, (
        f"Memory usage {peak_mem:.1f}MB exceeds {MAX_MEMORY_MB}MB limit"
    )


def test_prediction_latency(trained_model):
    """
    Test single prediction latency
    """
    start_time = time.perf_counter()
    trained_model.predict(SAMPLE_INPUT)
    latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"Prediction latency: {latency_ms:.1f}ms")

    assert latency_ms <= MAX_LATENCY_MS, (
        f"Prediction latency {latency_ms:.1f}ms exceeds {MAX_LATENCY_MS}ms limit"
    )


def test_throughput(trained_model):
    """
    Test predictions per second under load
    """
    n_runs = 1000
    start_time = time.perf_counter()
    
    for _ in range(n_runs):
        trained_model.predict(SAMPLE_INPUT)
    
    elapsed_time = time.perf_counter() - start_time
    throughput = n_runs / elapsed_time
    print(f"Throughput: {throughput:.1f} predictions/second")

    assert throughput >= MIN_THROUGHPUT, (
        f"Throughput {throughput:.1f} predictions/sec is below minimum {MIN_THROUGHPUT}"
    )
