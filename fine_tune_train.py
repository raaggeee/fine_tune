import os
import torch
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, TextStreamerfrom
import FastLanguageModel, is_bfloat16_supported


max_seq_length = 2048
model_name = ""
model, tokenizer = FastLanguageModel.from_pretrained(
    model=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj",
        "down_proj", "o_proj", "gate_proj"]
)

#we can also load our custom instruct dataset
dataset = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")

alpaca_template = """
Below is an instruction that describes a task. Write a response that appropriately completes the
request.

### Instruction:
{}

### Response:
{}
"""

EOS_TOKEN = tokenizer.eos_token

#might have to modify this
def format_samples(example):
    example["prompt"] = alpaca_template.format(example["instruction"], "")
    example["output"] = example["output"] + EOS_TOKEN
    return {
        "prompt": example["prompt"],
        "output": example["output"]
    }

dataset = dataset.map(format_samples, batched=True, remove_columns=dataset.column_name)

dataset = dataset.train_test_split(test_size=0.5)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
        save_strategy="steps",
        save_steps=100,

    )
)

trainer.train(resume_from_checkpoint=True)


## Test the model
FastLanguageModel.from_inference(model)
message = alpaca_template.format("What is QLora?", "")

input = tokenizer([message], return_tensor="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**input, streamer=text_streamer, max_new_tokens=256, use_cache=True)
