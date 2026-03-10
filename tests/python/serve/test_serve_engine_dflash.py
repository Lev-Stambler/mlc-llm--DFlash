# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals
"""Integration tests for DFlash speculative decoding engine.

These tests mirror the EAGLE tests in test_serve_engine_spec.py.
They require compiled models and will be skipped when models are not available.
"""
from typing import Callable, List, Optional

import numpy as np

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import Request, RequestStreamOutput, data
from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine
from mlc_llm.testing import require_test_model

prompts = [
    "What is the meaning of life?",
    "Introduce the history of Pittsburgh to me. Please elaborate in detail.",
    "Write a three-day Seattle travel plan. Please elaborate in detail.",
    "What is Alaska famous of? Please elaborate in detail.",
    "What is the difference between Lambda calculus and Turing machine? Please elaborate in detail.",
    "What are the necessary components to assemble a desktop computer? Please elaborate in detail.",
    "Why is Vitamin D important to human beings? Please elaborate in detail.",
    "Where is milk tea originated from? Please elaborate in detail.",
    "Where is the southernmost place in United States? Please elaborate in detail.",
    "Do you know AlphaGo? What capabilities does it have, and what achievements has it got? Please elaborate in detail.",
]


def create_requests(
    num_requests: int,
    stop_token_id: Optional[int] = None,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    max_tokens_low: int = 256,
    max_tokens_high: int = 257,
) -> List[Request]:
    assert num_requests >= 0 and num_requests <= len(prompts)

    stop_token_ids = [stop_token_id] if stop_token_id is not None else []
    requests = []
    for req_id, prompt in zip(range(num_requests), prompts):
        max_tokens = np.random.randint(max_tokens_low, max_tokens_high)
        requests.append(
            Request(
                request_id=str(req_id),
                inputs=data.TextData(prompt),
                generation_config=GenerationConfig(
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                ),
            )
        )
    return requests


@require_test_model("Qwen3-8B-q0f16-MLC")
def test_engine_dflash_basic(model: str):
    """Test DFlash engine without continuous batching.

    - Add all requests at once.
    - All requests have the same max_tokens.
    - Use DFlash speculative decoding mode.
    """

    num_requests = len(prompts)
    temperature = 0.9
    repetition_penalty = 1.0
    max_tokens: int = 256
    np.random.seed(0)

    outputs: List[List[int]] = [[] for _ in range(num_requests)]

    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            assert len(stream_outputs) == 1
            outputs[int(request_id)] += stream_outputs[0].delta_token_ids

    # DFlash draft model paths (adjust to match your compiled model layout)
    small_model = "dist/Qwen3-8B-DFlash-b16-q0f16-MLC"
    small_model_lib = (
        "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-MLC-cuda.so"
    )
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            speculative_mode="dflash",
            spec_draft_length=16,
        ),
        request_stream_callback=fcallback,
    )

    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
    )

    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max_tokens - 1
    for step in range(num_steps):
        engine.step()

    for req_id, output in enumerate(outputs):
        print(f"Prompt {req_id}: {requests[req_id].inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


@require_test_model("Qwen3-8B-q0f16-MLC")
def test_engine_dflash_continuous_batching(model: str):
    """Test DFlash engine with continuous batching.

    - All requests added at once.
    - Variable max_tokens per request.
    """

    num_requests = len(prompts)
    temperature = 0.9
    repetition_penalty = 1.00
    max_tokens_low = 128
    max_tokens_high = 384
    np.random.seed(0)

    outputs: List[List[int]] = [[] for _ in range(num_requests)]
    finish_time: List[Optional[int]] = [None] * num_requests

    class CallbackTimer:
        timer: int = -1

        def callback_getter(self) -> Callable[[List[RequestStreamOutput]], None]:
            def fcallback(delta_outputs: List[RequestStreamOutput]):
                for delta_output in delta_outputs:
                    request_id, stream_outputs = delta_output.unpack()
                    assert len(stream_outputs) == 1
                    if stream_outputs[0].finish_reason is not None:
                        print(f"Request {request_id} finished at step {self.timer}.")
                    outputs[int(request_id)] += stream_outputs[0].delta_token_ids
                    finish_time[int(request_id)] = self.timer

            return fcallback

        def step(self) -> None:
            self.timer += 1

    small_model = "dist/Qwen3-8B-DFlash-b16-q0f16-MLC"
    small_model_lib = (
        "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-MLC-cuda.so"
    )
    timer = CallbackTimer()
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            speculative_mode="dflash",
            spec_draft_length=16,
        ),
        request_stream_callback=timer.callback_getter(),
    )

    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens_low,
        max_tokens_high=max_tokens_high,
    )

    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max(request.generation_config.max_tokens for request in requests) - 1
    for step in range(num_steps):
        timer.step()
        assert timer.timer == step
        engine.step()

    for req_id, (request, output, fin_time) in enumerate(zip(requests, outputs, finish_time)):
        print(f"Prompt {req_id}: {request.inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


@require_test_model("Qwen3-8B-q0f16-MLC")
def test_engine_dflash_generate(model: str):
    """Test DFlash engine using the high-level .generate() API."""

    small_model = "dist/Qwen3-8B-DFlash-b16-q0f16-MLC"
    small_model_lib = (
        "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-MLC-cuda.so"
    )
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            speculative_mode="dflash",
            spec_draft_length=16,
        ),
    )

    num_requests = 10
    max_tokens = 256

    output_texts, _ = engine.generate(
        prompts[:num_requests], GenerationConfig(max_tokens=max_tokens, n=3)
    )
    for req_id, req_outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(req_outputs) == 1:
            print(f"Output {req_id}:{req_outputs[0]}\n")
        else:
            for i, output in enumerate(req_outputs):
                print(f"Output {req_id}({i}):{output}\n")


@require_test_model("Qwen3-8B-q0f16-MLC")
def test_engine_dflash_spec_efficiency(model: str):
    """Test DFlash speculative decoding efficiency metrics."""

    num_requests = 1
    temperature = 0.9
    repetition_penalty = 1.0
    max_tokens: int = 512
    np.random.seed(0)

    outputs: List[List[int]] = [[] for _ in range(num_requests)]

    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            assert len(stream_outputs) == 1
            outputs[int(request_id)] += stream_outputs[0].delta_token_ids

    small_model = "dist/Qwen3-8B-DFlash-b16-q0f16-MLC"
    small_model_lib = (
        "dist/Qwen3-8B-DFlash-b16-q0f16-MLC/Qwen3-8B-DFlash-b16-q0f16-MLC-cuda.so"
    )
    spec_engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(
            max_total_sequence_length=4096,
            additional_models=[(small_model, small_model_lib)],
            spec_draft_length=16,
            speculative_mode="dflash",
        ),
        request_stream_callback=fcallback,
    )

    requests = create_requests(
        num_requests,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
    )

    for request in requests:
        spec_engine.add_request(request)

    num_steps = num_requests + max_tokens - 1
    for step in range(num_steps):
        spec_engine.step()

    for eg, name in zip([spec_engine], ["DFlash Speculative Decoding"]):
        metrics = eg.metrics()
        print("engine name:", name)
        if "spec_decode" in metrics:
            print("spec decode:", metrics["spec_decode"])
        if "sum_num_draft_tokens" in metrics:
            print("total draft tokens:", metrics["sum_num_draft_tokens"])
        if "sum_num_accepted_tokens" in metrics:
            print("total accepted tokens:", metrics["sum_num_accepted_tokens"])
            print(
                "Accept rate:",
                metrics["sum_num_accepted_tokens"]
                / (1e-10 + metrics.get("sum_num_draft_tokens", 0)),
            )
        print("engine total decode time:", metrics["engine_decode_time_sum"])
        print()


if __name__ == "__main__":
    test_engine_dflash_basic()
    test_engine_dflash_continuous_batching()
    test_engine_dflash_generate()
    test_engine_dflash_spec_efficiency()
