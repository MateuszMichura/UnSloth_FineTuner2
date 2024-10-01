import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

def finetune_model(model, tokenizer, dataset, learning_rate, batch_size, num_epochs, weight_decay, warmup_steps, gradient_accumulation_steps, max_seq_length, packing):
    """
    Fine-tune a model on a given dataset using SFTTrainer.
    
    Args:
    model: The pre-trained model to fine-tune.
    tokenizer: The tokenizer associated with the model.
    dataset: The dataset to use for fine-tuning.
    learning_rate (float): Learning rate for optimization.
    batch_size (int): Number of training examples used in one iteration.
    num_epochs (int): Number of complete passes through the dataset.
    weight_decay (float): L2 regularization term.
    warmup_steps (int): Number of warmup steps for learning rate scheduler.
    gradient_accumulation_steps (int): Number of steps to accumulate gradients before performing a backward/update pass.
    max_seq_length (int): Maximum length of sequences after tokenization.
    packing (bool): Whether to enable prompt packing for efficiency.
    
    Returns:
    str: A message indicating the completion of training.
    """
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Calculate max_steps
    max_steps = (len(dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available() and not is_bfloat16_supported(),
        bf16=torch.cuda.is_available() and is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="linear",
        max_steps=max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=training_args,
        packing=packing,
    )

    trainer.train()
    
    print(f"Training completed successfully on {device}!")
    print(f"Model parameters are on: {next(model.parameters()).device}")
    
    return f"Training completed successfully on {device}!"