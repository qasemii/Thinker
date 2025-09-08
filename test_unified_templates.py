#!/usr/bin/env python3
"""Test script to verify unified chat template approach works correctly."""

import transformers

def test_unified_templates():
    """Test the unified chat template approach with different models."""
    
    # Test models
    test_models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-2b-it", 
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct"
    ]
    
    test_question = "What is the capital of France?"
    test_reasoning = "The capital of France is Paris, which is a major European city."
    
    for model_name in test_models:
        try:
            print(f"\nTesting model: {model_name}")
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.add_bos_token = False
            tokenizer.add_eos_token = False
            
            # Test evaluation template
            messages = [
                {"role": "user", "content": f"Answer the following question:\n### Question: {test_question}"}
            ]
            eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"Evaluation prompt: {eval_prompt[:100]}...")
            
            # Test training templates
            cr_messages = [
                {"role": "user", "content": f"Answer the following question:\n### Question: {test_question}"},
                {"role": "assistant", "content": f"### Answer: {test_reasoning}"}
            ]
            cr_prompt = tokenizer.apply_chat_template(cr_messages, tokenize=False)
            print(f"CR template: {cr_prompt[:100]}...")
            
            ir_messages = [
                {"role": "user", "content": f"Identify the incorrect options to reach the correct answers:\n### Question: {test_question}"},
                {"role": "assistant", "content": f"### Answer: {test_reasoning} Therefore the correct answer is option (A)."}
            ]
            ir_prompt = tokenizer.apply_chat_template(ir_messages, tokenize=False)
            print(f"IR template: {ir_prompt[:100]}...")
            
            print("✅ Success!")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")

if __name__ == "__main__":
    test_unified_templates()
