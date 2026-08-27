from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


text = "1+1等于多少？"

encoded = tokenizer(
    text,
    return_tensors="pt",
)


input_ids = encoded["input_ids"][0]


tokens = tokenizer.convert_ids_to_tokens(
    input_ids
)


print("Original:")
print(text)

print("\nToken IDs:")
print(input_ids)

print("\nTokens:")

for i, (token, token_id) in enumerate(
    zip(tokens, input_ids.tolist())
):
    print(
        f"{i:02d} "
        f"token={repr(token):20s} "
        f"id={token_id}"
    )


decoded = tokenizer.decode(
    input_ids
)

print("\nDecoded:")
print(decoded)