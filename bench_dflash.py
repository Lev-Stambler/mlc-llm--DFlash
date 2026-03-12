"""Benchmark DFlash speculative decoding vs normal decoding."""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    "Explain what a neural network is in simple terms.",
    "Write a short poem about the ocean.",
    "What are the three laws of thermodynamics?",
]

MAX_TOKENS = 128


_model_name_cache = {}

def _get_model_name(base_url):
    """Query /v1/models to discover the served model name."""
    if base_url in _model_name_cache:
        return _model_name_cache[base_url]
    models_url = base_url.rsplit("/v1/", 1)[0] + "/v1/models"
    with urllib.request.urlopen(models_url, timeout=10) as resp:
        data = json.loads(resp.read())
    name = data["data"][0]["id"]
    _model_name_cache[base_url] = name
    return name


def send_request(prompt, max_tokens=MAX_TOKENS, server_url=None):
    """Send a chat completion request and measure time + tokens."""
    url = server_url or "http://127.0.0.1:8000/v1/chat/completions"
    model_name = _get_model_name(url)
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    usage = body["usage"]
    content = body["choices"][0]["message"]["content"]
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "content": content,
    }


def warmup(server_url=None):
    """Send a warmup request to prime the model."""
    print("  Warming up...")
    send_request("Hi", max_tokens=8, server_url=server_url)
    print("  Warmup done.")


def run_benchmark(label, server_url=None):
    """Run benchmark suite and return results."""
    print(f"\n{'='*60}")
    print(f" Benchmarking: {label}")
    print(f"{'='*60}")
    warmup(server_url=server_url)

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i+1}: {prompt[:50]}...")
        r = send_request(prompt, server_url=server_url)
        tok_per_sec = r["completion_tokens"] / r["elapsed"]
        print(f"    {r['completion_tokens']} tokens in {r['elapsed']:.2f}s = {tok_per_sec:.1f} tok/s")
        print(f"    Output: {r['content'][:80]}...")
        results.append(r)
        total_tokens += r["completion_tokens"]
        total_time += r["elapsed"]

    avg_tps = total_tokens / total_time
    print(f"\n  TOTAL: {total_tokens} tokens in {total_time:.2f}s")
    print(f"  AVG THROUGHPUT: {avg_tps:.1f} tok/s")
    return {"label": label, "total_tokens": total_tokens, "total_time": total_time, "avg_tps": avg_tps, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MLC LLM server")
    parser.add_argument("label", nargs="?", default="unknown", help="Benchmark label")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    server_url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    result = run_benchmark(args.label, server_url=server_url)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved to: {args.output}")
