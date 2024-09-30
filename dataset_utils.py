from datasets import load_dataset, Dataset
import json
import csv
import openai
import anthropic
import requests
import os
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            dataset = load_dataset(dataset_path, split="train", use_auth_token=hf_token)
        except ValueError:
            # If use_auth_token is not supported, try without it
            dataset = load_dataset(dataset_path, split="train")
    elif dataset_source == 'local':
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")
        
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            elif isinstance(data, dict):
                dataset = Dataset.from_dict(data)
            else:
                raise ValueError("JSON file must contain either a list or a dictionary.")
        elif dataset_path.endswith('.csv'):
            with open(dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            dataset = Dataset.from_list(data)
        else:
            raise ValueError("Unsupported file format. Please use JSON or CSV.")
    else:
        raise ValueError("Invalid dataset source. Use 'huggingface' or 'local'.")

    # Check if 'conversations' column exists, if not, try to create it
    if 'conversations' not in dataset.column_names:
        if 'text' in dataset.column_names:
            dataset = dataset.map(lambda example: {'conversations': [{'human': example['text'], 'assistant': ''}]})
        else:
            raise ValueError("Dataset does not contain 'conversations' or 'text' column. Please check your dataset structure.")

    # Only apply standardize_sharegpt if 'conversations' column exists
    if 'conversations' in dataset.column_names:
        dataset = standardize_sharegpt(dataset)
    
    def formatting_prompts_func(examples):
        if tokenizer is None:
            raise ValueError("Tokenizer is not properly initialized. Please load the model and tokenizer before preparing the dataset.")
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    if 'text' not in dataset.column_names:
        def format_conversation(example):
            formatted_text = ""
            for turn in example['conversations']:
                formatted_text += f"{turn['role']}: {turn['content']}\n"
            return {"text": formatted_text.strip()}
        
        dataset = dataset.map(format_conversation)
    
    return dataset

def standardize_sharegpt(dataset):
    # This is a simplified version. You might need to adjust it based on your specific needs.
    def process_conversation(conversation):
        standardized = []
        for turn in conversation:
            if 'human' in turn:
                standardized.append({'role': 'user', 'content': turn['human']})
            if 'assistant' in turn:
                standardized.append({'role': 'assistant', 'content': turn['assistant']})
        return standardized

    return dataset.map(lambda x: {'conversations': process_conversation(x['conversations'])})

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
    You are an AI assistant creating training dataset for finetuning a model. 
    You are provided an one-shot or few-shot output example of output that application expects from the AI model. You are also provided the
    expected structure that the to-be trained AI model expects during training process.
    

    Examples:
    {examples}

    Expected structure:
    {expected_structure}

    Please help Generate a new dataset in the provided same style and expected structure. Do not produce any extra output except the dataset in the training needed structure:
    """
    
    if ai_provider == "OpenAI":
        client = openai.OpenAI(api_key=api_key)
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            try:
                response = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30  # 30 seconds timeout
                )
                conversation = response.choices[0].message.content
                synthetic_data.append({"conversations": json.loads(conversation)})
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode response as JSON: {response.choices[0].message.content}")
            except openai.APITimeoutError:
                logger.warning("OpenAI API request timed out")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
            time.sleep(1)  # Rate limiting
    
    elif ai_provider == "Anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            try:
                response = client.completions.create(
                    model="claude-3-opus-20240229",
                    prompt=f"Human: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=1000,
                    timeout=30  # 30 seconds timeout
                )
                synthetic_data.append({"conversations": json.loads(response.completion)})
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode response as JSON: {response.completion}")
            except anthropic.APITimeoutError:
                logger.warning("Anthropic API request timed out")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
            time.sleep(1)  # Rate limiting
    
    elif ai_provider == "Ollama":
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            try:
                response = requests.post('http://localhost:11434/api/generate',
                                         json={
                                             "model": model_name,
                                             "prompt": prompt,
                                             "stream": False
                                         },
                                         timeout=30)  # 30 seconds timeout
                response.raise_for_status()
                synthetic_data.append({"conversations": json.loads(response.json()["response"])})
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode response as JSON: {response.json()['response']}")
            except requests.Timeout:
                logger.warning("Ollama API request timed out")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
            time.sleep(1)  # Rate limiting
    
    dataset = Dataset.from_list(synthetic_data)
    dataset = standardize_sharegpt(dataset)
    
    if 'text' not in dataset.column_names:
        def format_conversation(example):
            formatted_text = ""
            for turn in example['conversations']:
                formatted_text += f"{turn['role']}: {turn['content']}\n"
            return {"text": formatted_text.strip()}
        
        dataset = dataset.map(format_conversation)
    
    return dataset