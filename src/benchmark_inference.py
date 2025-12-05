import time
import statistics

import numpy as np
import onnxruntime as ort


def benchmark(onnx_path: str = "artifacts/voice_net.onnx", iters: int = 100):
    """
    Simple ONNXRuntime latency benchmark.
    Measures average latency over N runs.
    """

    session = ort.InferenceSession(onnx_path)
    print(f"[INFO] Loaded ONNX model from {onnx_path}")

    # Dummy mel input
    mel = np.random.randn(1, 1, 64, 100).astype("float32")

    # Warmup
    for _ in range(10):
        _ = session.run(None, {"mel": mel})

    latencies = []

    for i in range(iters):
        start = time.time()
        _ = session.run(None, {"mel": mel})
        end = time.time()

        latency_ms = (end - start) * 1000.0
        latencies.append(latency_ms)

    avg = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    print(f"[RESULT] Average latency: {avg:.2f} ms over {iters} runs")
    print(f"[RESULT] P95 latency:   {p95:.2f} ms")


if __name__ == "__main__":
    benchmark()
