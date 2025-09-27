import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

model_name = "<Your Model Name>"
dataset_name = "<Your Train Dataset>"
SEED = "<Your SEED>"
DATASET_RATIO = "<Your DATASET_RATIO>"
LORA_RANK = "<Your LORA_RANK>"
MAX_COMPLETION_LEGNTH = 512

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.bfloat16()
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.shuffle(seed=SEED)
train_dataset = dataset.select(range(int(len(dataset) * DATASET_RATIO)))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)
model = get_peft_model(model, lora_config)
model = model.bfloat16()

dpo_config = DPOConfig(
    learning_rate=8e-6,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    dataloader_num_workers=4,
    max_completion_length=MAX_COMPLETION_LEGNTH,
    max_prompt_length=None,
    max_grad_norm=0.1,
    num_train_epochs=1,
    output_dir=f"./output_invthink_dpo_{model_name}",
    report_to="none",
    push_to_hub=False,
    save_strategy="steps",
    save_steps=100,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()
print("Training completed!")

final_model_path = f"./final_dpo_{model_name}"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Model saved to {final_model_path}")
