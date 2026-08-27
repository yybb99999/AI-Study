from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("BOS token:")
print(tokenizer.bos_token)

print("BOS token id:")
print(tokenizer.bos_token_id)

print("\nEOS token:")
print(tokenizer.eos_token)

print("EOS token id:")
print(tokenizer.eos_token_id)

print("\nPAD token:")
print(tokenizer.pad_token)

print("PAD token id:")
print(tokenizer.pad_token_id)

print("\nUNK token:")
print(tokenizer.unk_token)

print("UNK token id:")
print(tokenizer.unk_token_id)

print("\nAll special tokens:")
print(tokenizer.all_special_tokens)

print("\nAll special token ids:")
print(tokenizer.all_special_ids)