from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt
import openai
import anthropic
import requests
import json

def prepare_dataset(dataset_path, tokenizer):
    """
    Prepare a dataset for fine-tuning.
    
    Args:
    dataset_path (str): Path or identifier of the dataset.
    tokenizer: The tokenizer associated with the model.
    
    Returns:
    Dataset: Prepared dataset ready for fine-tuning.
    """
    dataset = load_dataset(dataset_path, split="train")
    dataset = standardize_sharegpt(dataset)
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def create_synthetic_dataset(examples, num_samples, ai_provider, api_key, model_name):
    """
    Create a synthetic dataset based on example conversations.
    
    Args:
    examples (str): Example conversations to base the synthetic data on.
    num_samples (int): Number of synthetic samples to generate.
    ai_provider (str): AI provider to use for generation ('OpenAI', 'Anthropic', or 'Ollama').
    api_key (str): API key for the chosen AI provider.
    model_name (str): Model name for Ollama (if applicable).
    
    Returns:
    Dataset: Synthetic dataset ready for fine-tuning.
    """
    synthetic_data = []
    
    if ai_provider == "OpenAI":
        openai.api_key = api_key
        for _ in range(num_samples):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant creating training data. Generate a conversation based on the given examples."},
                    {"role": "user", "content": f"Examples:\n{examples}\n\nGenerate a new conversation in the same style:"}
                ]
            )
            synthetic_data.append({"conversations": json.loads(response.choices[0].message.content)})
    
    elif ai_provider == "Anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        for _ in range(num_samples):
            response = client.completions.create(
                model="claude-2.1",
                prompt=f"Human: You are an AI assistant creating training data. Generate a conversation based on the given examples.\n\nExamples:\n{examples}\n\nGenerate a new conversation in the same style:\n\nAssistant: Here's a new conversation in the same style:\n\n",
                max_tokens_to_sample=300
            )
            synthetic_data.append({"conversations": json.loads(response.completion)})
    
    elif ai_provider == "Ollama":
        for _ in range(num_samples):
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         "model": model_name,
                                         "prompt": f"You are an AI assistant creating training data. Generate a conversation based on the given examples.\n\nExamples:\n{examples}\n\nGenerate a new conversation in the same style:",
                                         "stream": False
                                     })
            synthetic_data.append({"conversations": json.loads(response.json()["response"])})
    
    dataset = Dataset.from_list(synthetic_data)
    dataset = standardize_sharegpt(dataset)
    return dataset