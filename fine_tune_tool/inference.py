from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048
model_name = "unsloth/gemma-4-E2B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=None
)

FastLanguageModel.for_inference(model)

system_prompt = """
You are a helpful assistant.
"""
user_prompt = """
Given a user question you have to answer the user with your best knowledge.

<USER_QUERY>
{question}
</USER_QUERY>
"""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "For my morning exercise routine, it's important to know the air quality around my neighborhood. My address is 34.052235, -118.243683 in Los Angeles. Can you provide me with the air quality data for my location?"}
]

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4"
)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500)
