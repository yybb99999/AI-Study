from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

texts = [
    "你好",
    "你好，我正在学习大语言模型。",
    "Qwen is a large language model."
]

encoded = tokenizer(
    texts,
    padding=True,
    return_tensors="pt"
)

print("=" * 60)
print("input_ids")
print("=" * 60)
print(encoded["input_ids"])

print("\nShape:")
print(encoded["input_ids"].shape)

print("\n" + "=" * 60)
print("attention_mask")
print("=" * 60)
print(encoded["attention_mask"])