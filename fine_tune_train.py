import os

os.chdir("..")
os.environ["HF_HOME"] = os.path.abspath("huggingface_cache")

from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported


max_seq_length = 2048
model_name = "google/gemma-4-E2B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "v_proj"]
)

#we can also load our custom instruct dataset
dataset = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")

system_prompt = "You are a general question answerer to the user."
user_prompt = """
Given a user question you have to answer the user with your best knowledge.

<USER_QUERY>
{question}
</USER_QUERY>
"""

EOS_TOKEN = tokenizer.eos_token

#this is a preprocessing step. it will normalize the data into chat template (in our case gemma)
def format_samples(sample):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=sample["instruction"])},
            {"role": "assistant", "content": sample["output"]}
        ]
    }


dataset = dataset.map(format_samples, remove_columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=0.5)


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
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
