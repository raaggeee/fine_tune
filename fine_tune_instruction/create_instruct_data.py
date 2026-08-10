import concurrent.futures
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from typing import List, Tuple
from datasets import Dataset
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

def load_articles(file_path: str) -> Dataset:
    with open(file_path, "r") as f:
        data = json.load(f)

    return Dataset.from_dict({
        "id": [item["id"] for item in data["artifact_data"]],
        "content": [item["content"] for item in data["artifact_data"]],
        "author_id": [item["author_id"] for item in data["artifact_data"]],
        "author_full_name": [item["author_full_name"] for item in data["artifact_data"]],
        "link": [item["link"] for item in data["artifact_data"]],
    })

def clean_text(text):
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_substring(dataset, min_length = 1000, max_length = 2000):
    extracts = []
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    for article in dataset["content"]:
        cleaned_article = clean_text(article)
        sentences = re.split(sentence_pattern, cleaned_article)

        current_chunk = ""
        for sent in sentences:
            sent = sent.strip()

            if not sent:
                continue

            if len(current_chunk) + len(sent) <= max_length:
                current_chunk += sent + " "

            else:
                if len(current_chunk) > min_length:
                    extracts.append(current_chunk)

                current_chunk = " "

            if len(current_chunk) > min_length:
                extracts.append(current_chunk)

    return extracts

def generate_instruction_answer_pair(extract, client):
    #model depenedent
    prompt = f"""Based on the following extract, generate five
    instruction-answer pairs. Each instruction \
    must ask to write about a specific topic contained in the context.
    each answer \
    must provide a relevant paragraph based on the information found in
    the \
    context. Only use concepts from the context to generate the
    instructions. \
    Instructions must never explicitly mention a context, a system, a
    course, or an extract. \
    Instructions must be self-contained and general. \
    Answers must imitate the writing style of the context. \
    Example instruction: Explain the concept of an LLM Twin. \
    Example answer: An LLM Twin is essentially an AI character that
    mimics your writing style, personality, and voice. \
    It's designed to write just like you by incorporating these elements
    into a language model. \
    The idea is to create a digital replica of your writing habits using
    advanced AI techniques. \
    Provide your response in JSON format with the following structure:
    {{
    "instruction_answer_pairs": [
    {{"instruction": "...", "answer": "..."}},
    ...
    ]
    }}
    Extract:
    {extract}
    """
    return

class InstructionAnswerSet:
    def __init__(self, pairs):
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_st: str):
        data = json.loads(json_st)
        pairs = [(pair["instruction"], pair["answer"]) for pair in data["instruction_dataset"]]

        return cls(pairs)

    def __iter__(self):
        return iter(self.pairs)

def create_instruction_dataset(dataset, client, num_workers=4):
    extracts = extract_substring(dataset)
    instruction_answer_pair = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        #this sends the request to client to complete the API call
        for extract in extracts:
            futs = executor.submit(generate_instruction_answer_pair, extract, client)
            futures.append(futs)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            instruction_answer_pair.extend(future.result())

        instructions, answer = zip(*instruction_answer_pair)

        #change for hugging face
        # return Dataset.from_dict({
        #   "instruction": list(instructions),
        #   "answer": answer
        # })
        return {
            "instruction": instructions,
            "answer": answer
        }

def main():
    #to be changed
    client = None
    path_to_json = ""


    #1. Load Dataset
    raw_data = load_articles(path_to_json)
    print("Raw Data:")
    print(raw_data.to_pandas())

    instruction_data = create_instruction_dataset(raw_data, client)

    # filtered_data = instruction_data.train_test_split(test_size=0.1)
    filtered_data = instruction_data

    return filtered_data
