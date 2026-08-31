import torch
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "demo_sft.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen3_lora_demo"

MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():

    # ==========================================
    # 1. Tokenizer
    # ==========================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ==========================================
    # 2. Model
    # ==========================================

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    # ==========================================
    # 3. Dataset
    # ==========================================

    dataset = load_dataset(
        "json",
        data_files=str(DATA_PATH),
        split="train",
    )

    # ==========================================
    # 4. Format dataset
    # ==========================================

    def format_example(example):

        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text
        }

    dataset = dataset.map(
        format_example
    )

    print(dataset[0]["text"])

    # ==========================================
    # 5. LoRA
    # ==========================================

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ==========================================
    # 6. Training Config
    # ==========================================

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=False,
        fp16=False,
        max_length=256,
        report_to="none",
    )

    # ==========================================
    # 7. Trainer
    # ==========================================

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # ==========================================
    # 8. Train
    # ==========================================

    trainer.train()

    # ==========================================
    # 9. Save LoRA Adapter
    # ==========================================

    trainer.save_model(
        str(OUTPUT_DIR / "final")
    )


if __name__ == "__main__":
    main()