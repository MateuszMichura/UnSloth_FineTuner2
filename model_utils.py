import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path, hf_token):
    """
    Load a pre-trained model and tokenizer, using CUDA if available.
    
    Args:
    model_path (str): Path or identifier of the pre-trained model.
    hf_token (str): Hugging Face API token for accessing gated models.
    
    Returns:
    tuple: Loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            token=hf_token
        )
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.float32  # Use float32 for CPU
        )
    
    model.to(device)
    return model, tokenizer