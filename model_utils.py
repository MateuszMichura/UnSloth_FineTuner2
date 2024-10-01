import torch
import importlib.util
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path, hf_token):
    """
    Load a pre-trained model and tokenizer, using unsloth if available,
    falling back to standard transformers if necessary.
    
    Args:
    model_path (str): Path or identifier of the pre-trained model.
    hf_token (str): Hugging Face API token for accessing gated models.
    
    Returns:
    tuple: Loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"

    # Try to use unsloth if it's available
    if importlib.util.find_spec("unsloth") is not None:
        try:
            from unsloth import FastLanguageModel
            print("Using unsloth for model loading.")
            model, _ = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,  # Automatically choose between float16 and bfloat16
                load_in_4bit=cuda_available,  # Only use 4-bit quantization if CUDA is available
                token=hf_token
            )
        except Exception as e:
            print(f"Error loading model with unsloth: {e}")
            print("Falling back to standard transformers.")
            model = load_with_transformers(model_path, hf_token, device)
    else:
        print("unsloth not found. Using standard transformers.")
        model = load_with_transformers(model_path, hf_token, device)
    
    # Do not use .to(device) for quantized models
    # The device placement is handled automatically by unsloth or transformers
    
    return model, tokenizer

def load_with_transformers(model_path, hf_token, device):
    """Helper function to load model with standard transformers library."""
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # This will handle device placement automatically
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )