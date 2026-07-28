import os

os.chdir("..")
os.environ["HF_HOME"] = os.path.abspath("huggingface_cache")
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template


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
    target_modules=["q_proj"]
)

#we can also load our custom instruct dataset
dataset = load_dataset("json", data_files="fine_tune_tool/final_data.json")

system_prompt = """
You are a helpful assistant with access to a set of tools for making HTTP requests and testing APIs. You can use these tools to help users test endpoints, debug requests, retrieve data, and validate HTTP behavior.

## Available Tools

You have access to the following tools. For each tool, understand its purpose, required and optional parameters, and expected output format.

### Tool Usage Guidelines

1. **Understand the tool's purpose**: Before using a tool, clearly understand what it does and when it should be used.

2. **Check required parameters**: Each tool has required parameters (marked as "Required"). Ensure you have all necessary information before calling a tool.

3. **Handle optional parameters**: Optional parameters (marked as "Object" without "Required") can be omitted if not needed.

4. **Parse responses**: All tool outputs are in JSON format. Extract relevant information from the response structure and present it clearly to the user.

5. **Error handling**: If a tool call fails or returns an error status code, explain what went wrong and suggest next steps.

## Tool Description

<tool>
{TOOL_DESCRIPTIONS}
</tool>


Always prioritize the user's intent and provide helpful, accurate assistance.
"""
user_prompt = """
Given a user question you have to answer the user with your best knowledge.

<USER_QUERY>
{question}
</USER_QUERY>
"""

EOS_TOKEN = tokenizer.eos_token

#this is a preprocessing step. it will normalize the data into chat template (in our case gemma)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4"
)
def format_samples(sample):
    messages =  [
        {"role": "system", "content": system_prompt.format(TOOL_DESCRIPTIONS=sample["tool_desc"])},
        {"role": "user", "content": user_prompt.format(question=sample["instruction"])},
        {"role": "assistant", "content": sample["tool_reasoning"]},
        {"role": "assistant", "tool_calls": [{
            "type": "function",
            "function": {
                "name": sample["tool_name"],
                "arguments": sample["tool_input"]
            }
        }]},
        {"role": "tool", "name": sample["tool_name"], "content": sample["tool_result"]},
        {"role": "assistant", "content": sample["output"]}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


dataset = dataset.map(format_samples)

dataset = dataset.train_test_split(test_size=0.5)

sft_config = SFTConfig(
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
    max_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,
    dataloader_num_workers=0,
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    args=sft_config
)

trainer.train() #removed checkpoints
