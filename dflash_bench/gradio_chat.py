"""Simple Gradio chat UI that connects to MLC LLM's OpenAI-compatible API."""
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="none")


def chat_fn(message, history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="dflash-qwen3-8b",
        messages=messages,
        max_tokens=512,
        stream=True,
    )
    partial = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            partial += chunk.choices[0].delta.content
            yield partial


demo = gr.ChatInterface(
    chat_fn,
    title="DFlash Speculative Decoding - Qwen3-8B",
    description="q4f16_1 target + bf16 DFlash draft model. Check /metrics endpoint for acceptance rates.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
