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

"""Train the model sequentially: first CR prompt, then IR prompt."""

import argparse
import json
import os

import peft
import torch
import transformers
from utils import CastOutputToFloat

Dataset = torch.utils.data.Dataset


class CRDataset(Dataset):
  """Dataset for Chain of Reasoning (CR) training."""
  
  def __init__(self, ex, cr_template_func):
    self.data = ex
    self.cr_template_func = cr_template_func

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    
    # Use the gold answer reasoning for CR training
    cr_reasoning = self.cr_template_func(
        question=item['question'],
        reasoning=item['reasonings'][item['gold_answer']]
    )
    
    return cr_reasoning


class IRDataset(Dataset):
  """Dataset for Iterative Reasoning (IR) training."""
  
  def __init__(self, ex, ir_template_func):
    self.data = ex
    self.ir_template_func = ir_template_func

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    
    # Use all non-gold reasonings for IR training
    reasonings = dict(item['reasonings'])  # shallow copy
    reasonings.pop(item['gold_answer'], None)
    
    ir_reasoning = self.ir_template_func(
        question=item['question'],
        reasoning=" ".join(reasonings.values()),
        gold_answer=item['gold_answer']
    )
    
    return ir_reasoning


class BackwardDataCollator:
  """Collate the data for training."""

  def __init__(self, tokenizerr, label_pad_token_id=-100):
    self.tokenizerr = tokenizerr
    self.label_pad_token_id = label_pad_token_id

  def __call__(self, features):
    texts = features
    inputs = self.tokenizerr(texts,
                             padding=True,
                             truncation=True,
                             return_tensors='pt')
    inputs['labels'] = inputs['input_ids']
    return inputs


class BackwardTrainer(transformers.Trainer):
  """Trainer with standard single-input loss computation."""

  def compute_loss(self, model_instance, inputs, **kwargs):
    return model_instance(**inputs).loss


def train_phase(model, tokenizer, training_data, template_func, phase_name, args, epoch_offset=0):
  """Train the model for one phase (CR or IR)."""
  
  print(f"\n=== Starting {phase_name} Training Phase ===")
  
  # Create dataset and data collator
  if phase_name == "CR":
    dataset = CRDataset(training_data, template_func)
  else:  # IR
    dataset = IRDataset(training_data, template_func)
  
  data_collator = BackwardDataCollator(tokenizer)
  
  # Set learning rate based on model type
  lr = 5e-6 if 'mistral' in args.model else 2e-4
  
  # Training arguments
  training_args = transformers.TrainingArguments(
      output_dir=f'./outputs/{args.model}_{args.task}_{args.n}_{phase_name.lower()}',
      save_strategy='epoch',
      num_train_epochs=args.epochs_per_phase,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=lr,
      weight_decay=0.001,
      logging_dir=f'./logs/{phase_name.lower()}',
      logging_steps=100,
      remove_unused_columns=False,
      fp16=False,
      bf16=True,
      warmup_ratio=0.3,
      lr_scheduler_type='constant'
  )
  
  # Create trainer
  trainer = BackwardTrainer(
      model=model,
      args=training_args,
      train_dataset=dataset,
      data_collator=data_collator
  )
  
  # Train the model
  trainer.train()
  
  # Save the model after this phase
  save_path = f'./checkpoints/{args.model}_{args.task}_{args.n}_{phase_name.lower()}'
  os.makedirs(save_path, exist_ok=True)
  trainer.model.save_pretrained(save_path)
  print(f"Model saved to {save_path}")
  
  return trainer.model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', default=0, type=int, help='Percentage of data to use (0-100)')
  parser.add_argument('--task', default='ARC', type=str, help='Task name')
  parser.add_argument('--model', default='gemma-7b', type=str, help='Model name')
  parser.add_argument('--model_dir', default='', type=str, help='Model cache directory')
  parser.add_argument('--epochs_per_phase', default=5, type=int, help='Number of epochs per phase')
  parser.add_argument('--skip_cr', action='store_true', help='Skip CR training phase')
  parser.add_argument('--skip_ir', action='store_true', help='Skip IR training phase')
  parser.add_argument('--cr_checkpoint', default='', type=str, help='Path to CR checkpoint to load for IR phase')

  args = parser.parse_args()

  # Model selection
  if args.model == "mistral-7b":
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
  elif "Llama-3.2" in args.model:
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
  elif "gemma" in args.model:
    base_model = f"google/{args.model}-it"
  elif "qwen2.5" in args.model:
    base_model = f"Qwen/{args.model}-Instruct"
  elif args.model == 'olmo-2-1b':
    base_model = 'allenai/OLMo-2-0425-1B-Instruct'
  elif args.model == 'qwen-2.5-7b':
    base_model = 'Qwen/Qwen2.5-7B-Instruct'
  elif args.model == 'qwen-3-4b':
    base_model = 'Qwen/Qwen3-4B-Instruct-2507'
  else:
    raise ValueError(f"Unsupported model: {args.model}")

  # Load tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      base_model,
      model_max_length=1024,
      padding_side='right'
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.add_bos_token = False
  tokenizer.add_eos_token = False

  # Template functions
  def create_cr_template(question, reasoning):
    messages = [
      {"role": "user", "content": f"Answer the following question:\n### Question: {question}"},
      {"role": "assistant", "content": f"### Answer: {reasoning}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

  def create_ir_template(question, reasoning, gold_answer):
    messages = [
      {"role": "user", "content": f"Identify the incorrect options to reach the correct answers:\n### Question: {question}"},
      {"role": "assistant", "content": f"### Answer: {reasoning} Therefore the correct answer is option ({gold_answer})."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

  # Load training data
  teacher_data_file = f'./data/training_data/{args.task}.json'
  print(f"Loading data from {teacher_data_file}")
  with open(teacher_data_file, 'r') as f:
    data = json.load(f)

  num_samples = int(args.n / 100) * len(data)
  training_data = data[:num_samples]

  print(f'Using {args.n}% of data. ({num_samples}/{len(data)})')
  print(f'Training data size: {len(training_data)}')

  # Initialize model
  model = transformers.AutoModelForCausalLM.from_pretrained(
      base_model,
      device_map='auto',
      cache_dir=args.model_dir
  )

  model.config.use_cache = False
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model.lm_head = CastOutputToFloat(model.lm_head)

  lora_config = peft.LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=['q_proj', 'v_proj'],
      lora_dropout=0.05,
      bias='none',
      task_type='CAUSAL_LM'
  )

  model = peft.get_peft_model(model, lora_config)

  # Phase 1: CR Training
  if not args.skip_cr:
    model = train_phase(
        model=model,
        tokenizer=tokenizer,
        training_data=training_data,
        template_func=create_cr_template,
        phase_name="CR",
        args=args
    )
  else:
    print("Skipping CR training phase")

  # Phase 2: IR Training
  if not args.skip_ir:
    # Load CR checkpoint if specified
    if args.cr_checkpoint:
      print(f"Loading CR checkpoint from {args.cr_checkpoint}")
      model = peft.PeftModel.from_pretrained(model, args.cr_checkpoint)
    
    model = train_phase(
        model=model,
        tokenizer=tokenizer,
        training_data=training_data,
        template_func=create_ir_template,
        phase_name="IR",
        args=args
    )
  else:
    print("Skipping IR training phase")

  # Save final model
  final_save_path = f'./checkpoints/{args.model}_{args.task}_{args.n}_final'
  os.makedirs(final_save_path, exist_ok=True)
  model.save_pretrained(final_save_path)
  print(f"Final model saved to {final_save_path}")
  
  print("\n=== Training Complete ===")
  print(f"CR checkpoint: ./checkpoints/{args.model}_{args.task}_{args.n}_cr")
  print(f"IR checkpoint: ./checkpoints/{args.model}_{args.task}_{args.n}_ir")
  print(f"Final checkpoint: {final_save_path}")
