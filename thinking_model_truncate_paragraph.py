# %%
import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint
from datasets import load_dataset
import copy
import re
import pickle


# model_name = "Qwen/Qwen2.5-1.5B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %%
## Load the model
# model = HookedTransformer.from_pretrained(model_name, device=device)
# model.eval()
# Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model_tf = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=t.float16)
model_tf.eval()  # Set model to evaluation mode

# %%
def format_prompt(question):
    return f"Question: {question}\nRespond: <think>\n"

# %%
def find_tensor_divergence_point(tensor1, tensor2):
    """
    Compares two 1D integer tensors and returns the number of identical values
    from the beginning until they diverge.
    
    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        
    Returns:
        int: Number of matching values from the beginning
    """
    # Find the minimum length between the two tensors
    min_length = min(tensor1.size(0), tensor2.size(0))
    
    # Create a boolean tensor indicating which elements are equal
    equality = tensor1[:min_length] == tensor2[:min_length]
    
    # Find the first False value, which is where tensors diverge
    # If all values match up to min_length, nonzero() will return empty tensor
    divergence_points = (equality == False).nonzero(as_tuple=True)[0]
    
    # If they're identical up to min_length, return min_length
    if len(divergence_points) == 0:
        return min_length
    
    # Otherwise, return the index of the first divergence
    return divergence_points[0].item()

# %%
def count_steps(text):
    pattern = r"Step \d+:"
    # Find all matches (case insensitive)
    matches = re.findall(pattern, text)
    return len(matches)

# %%
def generate_with_cot(question, temperature, max_new_tokens = None, max_length  = None, formatted = False):
    assert max_new_tokens is not None or max_length is not None, "Either max_new_tokens or max_length should be provided."
    assert max_new_tokens is None or max_length is None, "Only one of max_new_tokens or max_length should be provided."

    if not formatted:
        question = format_prompt(question)
    inputs = tokenizer(question, return_tensors="pt").to(model_tf.device)
    
    # Generate response with chain-of-thought
    t.manual_seed(0)
    if max_new_tokens is not None:
        generated_tokens = model_tf.generate(
            **inputs, 
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # do_sample=False,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        generated_tokens = model_tf.generate(
            **inputs, 
            temperature=temperature,
            max_length=max_length,
            # do_sample=False,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_list = []
    for token in generated_tokens:
        if token[0].item() == tokenizer.bos_token_id:
            token = token[1:]
        response = tokenizer.decode(token)
        response_list.append(response)
    
    # response_truncated_list = removing_extra_text(response_list, stop_sequences)
    # step_count = [count_steps(response) for response in response_truncated_list]
    return response_list, generated_tokens, inputs["input_ids"].shape[-1], generated_tokens.shape[-1]

    # response = tokenizer.decode(generated_tokens[0])
    # return response

# %%
def extract_last_integer_efficient(text):
    # Pattern that matches both positive and negative numbers with optional commas
    # pattern = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)'
    pattern = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    
    # Search from the end of the string
    return re.findall(pattern, text)[-1].replace(',', '')

# %%
def string_to_number(string_value):
    # Remove commas and convert to int or float
    clean_string = string_value.replace(',', '')
    
    # Check if it's an integer or float
    if '.' in clean_string:
        return float(clean_string)
    else:
        return int(clean_string)

# %%
temperature = 0.6
max_new_tokens=1200
# stop_sequences = ["<|endoftext|>"]
tag = "Respond: <think>"

## Load the dataset
ds = load_dataset("gsm8k", "main")
test_data = ds["test"]  # Use the test split
# question = ds["test"][0]["question"]

# ds = load_dataset("HuggingFaceH4/MATH-500")
# test_data = ds["test"]
# question = ds["test"][0]["problem"]

# %%
answer_true = [extract_last_integer_efficient(ans) for ans in test_data["answer"]]

# %%
answer_dict = {}

data_random_number = np.random.randint(low=0, high=len(test_data), size=20).tolist()

for data_i in data_random_number:
    data = test_data[data_i]
    # question = data["problem"]
    question = data["question"]
    answer_true_i = answer_true[data_i]

    print(f"Data index: {data_i}, True answer: {answer_true_i}")

    response_list, generated_tokens, input_len, max_len = generate_with_cot(
        question, temperature, max_new_tokens = max_new_tokens
    )

    response_base = response_list[0]
    if generated_tokens[0,-1] == tokenizer.eos_token_id:
        answer_base = extract_last_integer_efficient(response_base)
    else:
        answer_base = None
        continue

    # tokens_based = generated_tokens[0]
    # if tokens_based[0].item() == tokenizer.bos_token_id:
    #     tokens_based = tokens_based[1:]

    answer_list = []

    index = response_base.find(tag)
    lines_before_tag = response_base[:index + len(tag)]
    lines_after_tag = response_base[index + len(tag):].split('\n')

    paragraph_indices = [i for i, item in enumerate(lines_after_tag) if item == "" and i != 0]

    for i_para in paragraph_indices:
        modified_text = lines_before_tag + '\n'.join(lines_after_tag[:i_para+1]) + '\n'
        modified_list, generated_tokens_2, input_len_2, max_len_2 = generate_with_cot(
            modified_text, temperature, max_length = input_len + max_new_tokens + 1, formatted = True
        )
        if generated_tokens_2[0,-1] == tokenizer.eos_token_id:
            answer_modified = extract_last_integer_efficient(modified_list[0])
        else:
            answer_modified = None
        answer_list.append([answer_true_i, answer_base, i_para, answer_modified, input_len, max_len, max_len_2])
        print(f"\t\t{answer_list[-1]}")
            
    answer_dict[data_i] = answer_list

with open("data/answer_dict_truncate_paragraph.pkl", "wb") as f:
    pickle.dump(answer_dict, f)
