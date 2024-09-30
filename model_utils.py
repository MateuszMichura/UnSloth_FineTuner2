from unsloth import FastLanguageModel

def load_model(model_path, hf_token):
    """
    Load a pre-trained model and tokenizer.
    
    Args:
    model_path (str): Path or identifier of the pre-trained model.
    hf_token (str): Hugging Face API token for accessing gated models.
    
    Returns:
    tuple: Loaded model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,  # Maximum sequence length the model can handle
        dtype=None,  # Automatically detect the appropriate data type
        load_in_4bit=True,  # Use 4-bit quantization to reduce memory usage
        token=hf_token
    )
    return model, tokenizer