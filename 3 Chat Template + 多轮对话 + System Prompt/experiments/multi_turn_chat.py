import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model.eval()

messages = [
    {
        "role": "system",
        "content": "你是一名AI学习助手，请使用简洁语言回答。"
    }
]

while True:

    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    prompt_length = model_inputs["input_ids"].shape[1]

    output_ids = generated_ids[0][prompt_length:]

    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )

    print("\nAssistant:", response)

    messages.append({
        "role": "assistant",
        "content": response
    })

    print(
        "Current context tokens:",
        model_inputs["input_ids"].shape[1]
    )