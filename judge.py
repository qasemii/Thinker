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
from tqdm import tqdm




reasoning_prompt="""
        You are given a question and its reasoning text. The reasoning text may contain one or more references to answer choices.
        Your task is to extract the final correct answer choice from the reasoning text.
        
        Follow these rules strictly:
        
        1. If the text indicates exactly one unique answer option (e.g., A, B, C, D), output that option letter only.
        
        2. If the text contains multiple different answer options marked as correct, output "N/A".
        
        3. If the text contains no clear answer option, output "N/A".
        
        Do not explain your choice. Output only the final label.

        ### Question: {question}
                
        ### Reasoning text: {reasoning}
        
        ### Answer:
"""

# reasoning_prompt= """Based on a given question and a reasoning, if there is a single  the answer from the reasoning. Your extracted answer is either a single letter (A-Z), "N/A" (if there is no correct answer detected), or "Multiple" (if there are multiple correct answers detected). Return the answer without any explanation.\n
#                      ### Question: {question}\n
#                      ### Reasoning: {reasoning}\n
#                      ### Answer:"""

def generate_reasoning(prompt, client, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):

    response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": "You are a helpful assistant that excels at reasoning."},
              {"role": "user", "content": prompt}
          ],
          temperature=0.0,
          max_tokens=10,
      )

    reasoning = response.choices[0].message.content.strip()
    
    return reasoning


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default="", type=str)
  parser.add_argument('--max_examples', default=None, type=int)
  args = parser.parse_args()

  api_key = "tgp_v1_Hlyiw1xzK5t5UP_If943fjZI7GAbGDHySHvMRv-rKg8"  # Reza
  client = OpenAI(
        api_key = api_key,
        base_url="https://api.together.xyz/v1",
  )

  with open(args.data_dir, "r") as f:
    results = json.load(f)
  
  if args.max_examples:
    results = results[:args.max_examples]


  count=0
  # forward reasoning generation
  judged = []
  for i, example in enumerate(tqdm(results, desc="Generating reasoning ...")):
    tmp = {}
    tmp["question"] = example["question"]
    tmp["gold_answer"] = example["gold_answer"]
    tmp["reasoning"] = example["reasoning"]
            
    generation_prompt = reasoning_prompt.format(question=tmp["question"],
                                                reasoning=tmp["reasoning"])
    tmp["pred"] = generate_reasoning(generation_prompt, client)
        
    judged.append(tmp)
    
    if tmp["pred"]==tmp["gold_answer"]:
        count+=1
  print("Accuracy: ", count/len(results))
  with open(f"./judge.json", "w") as f:
    json.dump(judged, f)
