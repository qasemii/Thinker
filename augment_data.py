# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Augment dataset with forward reasoning."""

import argparse
import json
from openai import OpenAI

from prompt import options
from prompt import reasoning_prompt

from tqdm import tqdm
from utils import get_alphabet_choice
from math_utils import parse_math_boxed
from math_utils import parse_number

def generate_reasoning(prompt, client, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):

    response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": "You are a helpful assistant that excels at explaining commonsense reasoning."},
              {"role": "user", "content": prompt}
          ],
          temperature=0.8,
          max_tokens=1000,
      )

    reasoning = response.choices[0].message.content.strip()
    
    return reasoning


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="SQA", type=str)
  parser.add_argument('--max_examples', default=None, type=int)
  args = parser.parse_args()

  api_key = "tgp_v1_Hlyiw1xzK5t5UP_If943fjZI7GAbGDHySHvMRv-rKg8"  # Reza
  client = OpenAI(
        api_key = api_key,
        base_url="https://api.together.xyz/v1",
  )

  with open(f"./data/training_data/{args.task}.json", "r") as f:
    dataset = json.load(f)
  
  if args.max_examples:
    dataset = dataset[:args.max_examples]

  is_math = False
  if args.task == "SQA":
    answer_extraction = get_yes_no
  elif args.task in ["ANLI", "ARC", "Date", "CSQA", "ESNLI"]:
    answer_extraction = get_alphabet_choice
  elif args.task in ["GSM8K", "GSM8K-Rev"]:
    answer_extraction = parse_number
    is_math = True
  elif args.task in ["TabMWP", "MATH"]:
    answer_extraction = parse_math_boxed
    is_math = True
  else:
    raise ValueError(f"Unsupported task: {args.task}")


  # forward reasoning generation
  results = []
  for i, example in enumerate(tqdm(dataset, desc="Generating reasoning ...")):
    tmp = {}
    tmp["question"] = example["question"]
    tmp["gold_answer"] = example["gold_answer"]
    
    tmp["reasonings"] = {}
        
    # reasoning for incorrect answers
    for o in options[args.task]:
        is_correct = "correct" if o == tmp["gold_answer"] else "incorrect"
        generation_prompt = example["question"] + reasoning_prompt.format(option=o, is_correct=is_correct)
        reasoning = generate_reasoning(generation_prompt, client)
        tmp["reasonings"][o] = reasoning
        
    results.append(tmp)

  
  with open(f"./data/training_data/{args.task}.json", "w") as f:
    json.dump(results, f)
