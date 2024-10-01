from huggingface_hub import HfApi
import os

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
        model.save_pretrained(temp_dir)
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