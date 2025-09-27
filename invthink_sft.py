from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

model_name = "<Your Model Name>"
dataset_name = "<Your Train Dataset>"

model = AutoModelForCausalLM.from_pretrained(model_name)
train_dataset = load_dataset(dataset_name, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def to_messages(example):
    """
    This example assumes that your dataset consists of 3 components (1. query , 2. inverse reasoning and 3. safe response)
    """
    messages = example["messages"]
    query = messages[0]["content"]
    inverse_reasoning = messages[1]["content"]
    safe_response = messages[2]["content"]

    return {
        "input_prompt": [
            {"role": "user", "content": query.strip()},
            {
                "role": "assistant",
                "content": inverse_reasoning + "\n" + safe_response.strip(),
            },
        ]
    }


train_dataset = train_dataset.map(to_messages, batched=False)


def formatting_function(examples):
    return tokenizer.apply_chat_template(
        examples["input_prompt"], tokenize=False, add_generation_prompt=False
    )


training_args_dict = {
    "output_dir": f"./output_invthink_{model_name}",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 6,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 5,
    "report_to": "none",
    "remove_unused_columns": False,
    "push_to_hub": False,
    "fp16": True,
}
training_args = TrainingArguments(**training_args_dict)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    formatting_func=formatting_function,
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()
print("Training completed!")

final_model_path = f"./final_sft_{model_name}"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Model saved to {final_model_path}")
