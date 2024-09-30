def convert_to_gguf(model, tokenizer, output_path, quantization_method="q8_0"):
    """
    Convert the fine-tuned model to GGUF format.
    
    Args:
    model: The fine-tuned model to convert.
    tokenizer: The tokenizer associated with the model.
    output_path (str): The path to save the converted model.
    quantization_method (str): The quantization method to use (e.g., "q8_0", "q4_k_m", "q5_k_m", "f16").
    
    Returns:
    str: A message indicating the success or failure of the conversion.
    """
    try:
        model.save_pretrained_gguf(output_path, tokenizer, quantization_method=quantization_method)
        return f"Model successfully converted to GGUF format: {output_path}-unsloth-{quantization_method}.gguf"
    except Exception as e:
        return f"Error converting to GGUF: {str(e)}"