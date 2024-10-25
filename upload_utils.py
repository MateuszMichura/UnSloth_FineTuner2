from huggingface_hub import HfApi
import os
import torch
from safetensors.torch import save_file  # Używane, jeśli chcesz użyć safetensors

def upload_to_huggingface(model, tokenizer, repo_name, token):
    """
    Upload a fine-tuned model and tokenizer to Hugging Face.
    
    Args:
    model: The fine-tuned model to upload.
    tokenizer: The tokenizer associated with the model.
    repo_name (str): The name of the repository to create/update on Hugging Face.
    token (str): Hugging Face API token.
    
    Returns:
    str: A message indicating the success or failure of the upload.
    """
    try:
        # Save the model and tokenizer to a temporary directory
        temp_dir = "temp_model"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Zapisz model przy użyciu torch.save lub safetensors
        # Zakomentuj jedną z poniższych linii w zależności od tego, której metody chcesz użyć:

        # Metoda torch.save
        torch.save(model.state_dict(), f"{temp_dir}/pytorch_model.bin")
        
        # Metoda safetensors
        # save_file(model.state_dict(), f"{temp_dir}/model.safetensors")

        tokenizer.save_pretrained(temp_dir)

        # Initialize the Hugging Face API
        api = HfApi()

        # Create or update the repository
        api.create_repo(repo_id=repo_name, token=token, exist_ok=True)

        # Upload the model and tokenizer files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            token=token
        )

        # Clean up the temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        return f"Model successfully uploaded to https://huggingface.co/{repo_name}"
    except Exception as e:
        return f"Error uploading model: {str(e)}"
    

def upload_gguf_to_huggingface(gguf_file_path, repo_name, token):
    """
    Upload a GGUF converted model to Hugging Face.
    
    Args:
    gguf_file_path (str): The path to the GGUF file.
    repo_name (str): The name of the repository to create/update on Hugging Face.
    token (str): Hugging Face API token.
    
    Returns:
    str: A message indicating the success or failure of the upload.
    """
    try:
        if not os.path.exists(gguf_file_path):
            return f"Error: GGUF file not found at {gguf_file_path}"

        # Initialize the Hugging Face API
        api = HfApi()

        # Create or update the repository
        api.create_repo(repo_id=repo_name, token=token, exist_ok=True)

        # Upload the GGUF file
        api.upload_file(
            path_or_fileobj=gguf_file_path,
            path_in_repo=os.path.basename(gguf_file_path),
            repo_id=repo_name,
            token=token
        )

        return f"GGUF model successfully uploaded to https://huggingface.co/{repo_name}"
    except Exception as e:
        return f"Error uploading GGUF model: {str(e)}"
