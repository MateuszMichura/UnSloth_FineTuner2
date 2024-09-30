from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt
import json
import csv

def prepare_dataset(dataset_source, dataset_path, tokenizer):
    """
    Prepare a dataset for fine-tuning, either from Hugging Face or a local file.
    
    Args:
    dataset_source (str): 'huggingface' or 'local'
    dataset_path (str): Path or identifier of the dataset
    tokenizer: The tokenizer associated with the model
    
    Returns:
    Dataset: Prepared dataset ready for fine-tuning
    """
    if dataset_source == 'huggingface':
        dataset = load_dataset(dataset_path, split="train")
    elif dataset_source == 'local':
        # Determine file type and load accordingly
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