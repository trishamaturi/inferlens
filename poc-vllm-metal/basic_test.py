"""
Phase 0 smoke test: prove we can talk to vLLM, via offline batched
inference (no server, no OTLP/metrics endpoint yet).

Demonstrates the two distinct modes:
  - .generate(): a batch of independent raw-text prompts, no chat
    template -- the model just continues the string. Classic vLLM
    quickstart style ("Hello, my name is" -> model completes it).
  - .chat(): a "conversation" (list of role-tagged turns) run through
    the model's chat template -- what an instruct-tuned model actually
    expects for question-answering.

Run inside the vllm-metal venv:
    source .venv-vllm-metal/bin/activate
    python hello_vllm.py
"""

from vllm import LLM, SamplingParams

# Small enough to run comfortably in 16GB of unified memory on an M4.
MODEL = "Qwen/Qwen3-0.6B"

RAW_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

CONVERSATIONS = [
    [{"role": "user", "content": "In one sentence, what does a KV cache do in LLM inference?"}],
    [{"role": "user", "content": "What's the tallest mountain in the world?"}],
    [{"role": "user", "content": "Write a haiku about rain."}],
    [
        {"role": "user", "content": "What's a good name for a pet fox?"},
        {"role": "assistant", "content": "How about Ember?"},
        {"role": "user", "content": "I like it. Give me one more option."},
    ],
]


def main() -> None:
    llm = LLM(model=MODEL)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=64)

    print("\n### Batched raw completions (.generate(), no chat template) ###")
    for output in llm.generate(RAW_PROMPTS, sampling_params):
        print("=" * 60)
        print(f"Prompt:     {output.prompt!r}")
        print(f"Completion: {output.outputs[0].text!r}")

    print("\n### Conversation (.chat(), chat template applied) ###")
    # .chat() applies the model's chat template instead of treating the
    # prompt as raw text completion -- matters for instruct-tuned models.
    chat_outputs = llm.chat(
        CONVERSATIONS, sampling_params, chat_template_kwargs={"enable_thinking": False}
    )
    for output in chat_outputs:
        print("=" * 60)
        print(f"Prompt:     {output.prompt!r}")
        print(f"Completion: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
