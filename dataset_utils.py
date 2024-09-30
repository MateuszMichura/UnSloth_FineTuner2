from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt
import json
import csv
import openai
import anthropic
import requests
import os

def prepare_dataset(dataset_source, dataset_path, tokenizer, hf_token=None):
    """
    Prepare a dataset for fine-tuning, either from Hugging Face or a local file.
    
    Args:
    dataset_source (str): 'huggingface' or 'local'
    dataset_path (str): Path or identifier of the dataset
    tokenizer: The tokenizer associated with the model
    hf_token (str, optional): Hugging Face token for accessing datasets
    
    Returns:
    Dataset: Prepared dataset ready for fine-tuning
    """
    if dataset_source == 'huggingface':
        dataset = load_dataset(dataset_path, split="train", use_auth_token=hf_token)
    elif dataset_source == 'local':
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")
        
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_dict(data)
        elif dataset_path.endswith('.csv'):
            with open(dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            dataset = Dataset.from_list(data)
        else:
            raise ValueError("Unsupported file format. Please use JSON or CSV.")
    else:
        raise ValueError("Invalid dataset source. Use 'huggingface' or 'local'.")

    dataset = standardize_sharegpt(dataset)
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def create_synthetic_dataset(examples, expected_structure, num_samples, ai_provider, api_key, model_name=None):
    """
    Create a synthetic dataset based on example conversations and expected structure.
    
    Args:
    examples (str): Example conversations to base the synthetic data on.
    expected_structure (str): Description of the expected dataset structure.
    num_samples (int): Number of synthetic samples to generate.
    ai_provider (str): AI provider to use for generation ('OpenAI', 'Anthropic', or 'Ollama').
    api_key (str): API key for the chosen AI provider.
    model_name (str, optional): Model name for Ollama (if applicable).
    
    Returns:
    Dataset: Synthetic dataset ready for fine-tuning.
    """
    synthetic_data = []
    
    prompt = f"""
    You are an AI assistant creating training data. Generate a conversation based on the given examples and structure.

    Examples:
    {examples}

    Expected structure:
    {expected_structure}

    Generate a new conversation in the same style and structure:
    """
    
    if ai_provider == "OpenAI":
        openai.api_key = api_key
        for _ in range(num_samples):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            synthetic_data.append({"conversations": json.loads(response.choices[0].message.content)})
    
    elif ai_provider == "Anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        for _ in range(num_samples):
            response = client.completions.create(
                model="claude-2.1",
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=1000
            )
            synthetic_data.append({"conversations": json.loads(response.completion)})
    
    elif ai_provider == "Ollama":
        for _ in range(num_samples):
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         "model": model_name,
                                         "prompt": prompt,
                                         "stream": False
                                     })
            synthetic_data.append({"conversations": json.loads(response.json()["response"])})
    
    dataset = Dataset.from_list(synthetic_data)
    dataset = standardize_sharegpt(dataset)
    return dataset