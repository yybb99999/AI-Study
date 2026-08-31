import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():

    print("=" * 60)
    print("1. Load tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")


    print("\n" + "=" * 60)
    print("2. Load model")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.eval()

    print("Model loaded.")
    print("Device:", next(model.parameters()).device)


    prompt = "1+1等于多少？"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]


    print("\n" + "=" * 60)
    print("3. Messages")
    print("=" * 60)

    print(messages)


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    print("\n" + "=" * 60)
    print("4. Chat Template")
    print("=" * 60)

    print(text)


    model_inputs = tokenizer(
        text,
        return_tensors="pt",
    )


    print("\n" + "=" * 60)
    print("5. input_ids")
    print("=" * 60)

    print(model_inputs["input_ids"])

    print("\nShape:")
    print(model_inputs["input_ids"].shape)


    print("\n" + "=" * 60)
    print("6. Generation")
    print("=" * 60)

    with torch.no_grad():

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32,
            do_sample=False,
        )


    print("Generated IDs shape:")
    print(generated_ids.shape)


    prompt_length = model_inputs["input_ids"].shape[1]

    output_ids = generated_ids[0][prompt_length:]


    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )


    print("\n" + "=" * 60)
    print("7. Response")
    print("=" * 60)

    print(response)


if __name__ == "__main__":
    main()