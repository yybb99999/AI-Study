from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "user",
        "content": "请解释什么是Transformer。"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=" * 60)
print("Original messages")
print("=" * 60)
print(messages)

print("\n" + "=" * 60)
print("After chat template")
print("=" * 60)
print(text)

text_true = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

text_false = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print("=== True ===")
print(text_true)

print("\n=== False ===")
print(text_false)