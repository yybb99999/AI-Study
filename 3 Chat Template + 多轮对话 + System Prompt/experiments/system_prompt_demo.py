from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "system",
        "content": "你是一名计算机专业教师，请使用简洁、准确的语言回答。"
    },
    {
        "role": "user",
        "content": "什么是Transformer？"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(text)