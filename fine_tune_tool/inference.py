from unsloth import FastLanguageModel
from transformers import TextStreamer


max_seq_length = 2048
model_name = "unsloth/gemma-4-E2B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False
)

FastLanguageModel.for_inference(model)

inputs = tokenizer("Hey!", return_tensors="pt").to("cuda")

text_streamer = TextStreamer(tokenizer)

print(model.generate(**inputs, streamer=text_streamer, max_new_tokens=100))
