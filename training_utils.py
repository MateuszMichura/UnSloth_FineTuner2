import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def finetune_model(model, tokenizer, dataset, learning_rate, batch_size, num_epochs):
    """
    Fine-tune a model on a given dataset, using CUDA if available.
    
    Args:
    model: The pre-trained model to fine-tune.
    tokenizer: The tokenizer associated with the model.
    dataset: The dataset to use for fine-tuning.
    learning_rate (float): Learning rate for optimization.
    batch_size (int): Number of training examples used in one iteration.
    num_epochs (int): Number of complete passes through the dataset.
    
    Returns:
    str: A message indicating the completion of training.
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),  # Use mixed precision training if CUDA is available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()
    return f"Training completed successfully on {device}!"