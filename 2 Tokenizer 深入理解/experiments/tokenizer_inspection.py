from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("=" * 60)
print("Tokenizer basic information")
print("=" * 60)

print("Tokenizer class:")
print(type(tokenizer))

print("\nVocabulary size:")
print(tokenizer.vocab_size)

print("\nLength of tokenizer:")
print(len(tokenizer))

texts = [
    "Hello world",
    "你好世界",
    "machine learning",
    "机器学习",
    "Qwen3",
    "large language model",
]

for text in texts:
    print("\n" + "=" * 60)
    print("Text:", text)

    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("Token IDs:")
    print(ids)

    print("Tokens:")
    print(tokens)

    print("Token count:")
    print(len(ids))