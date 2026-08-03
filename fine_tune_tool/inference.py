from unsloth import FastLanguageModel
from transformers import TextStreamer


max_seq_length = 2048
model_name = "unsloth/gemma-4-E2B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=None
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a very helpful assistant."},
    {"role": "user", "content": "Hey!"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=100)
