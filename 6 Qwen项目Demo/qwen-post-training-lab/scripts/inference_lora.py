from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-0.6B"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "qwen3_lora_demo"
    / "final"
)


def generate(model, tokenizer, prompt):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    prompt_length = (
        model_inputs["input_ids"].shape[1]
    )

    with torch.no_grad():

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    output_ids = generated_ids[0][
        prompt_length:
    ]

    return tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )


def main():

    print("=" * 60)
    print("Adapter path:")
    print(ADAPTER_PATH)

    print(
        "Adapter directory exists:",
        ADAPTER_PATH.exists()
    )

    print(
        "adapter_config.json exists:",
        (
            ADAPTER_PATH
            / "adapter_config.json"
        ).exists()
    )

    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
        )
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
    )

    model.eval()

    prompt = "什么是机器学习？"

    response = generate(
        model,
        tokenizer,
        prompt,
    )

    print("\nPrompt:")
    print(prompt)

    print("\nLoRA-SFT Response:")
    print(response)


if __name__ == "__main__":
    main()