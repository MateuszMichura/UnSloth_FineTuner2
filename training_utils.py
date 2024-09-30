from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

def finetune_model(model, tokenizer, dataset, learning_rate, batch_size, num_epochs):
    """
    Fine-tune a model on a given dataset.
    
    Args:
    model: The pre-trained model to fine-tune.
    tokenizer: The tokenizer associated with the model.
    dataset: The dataset to use for fine-tuning.
    learning_rate (float): Learning rate for optimization. Controls how quickly the model updates its weights.
    batch_size (int): Number of training examples used in one iteration. Larger batch sizes can lead to faster training but require more memory.
    num_epochs (int): Number of complete passes through the dataset. More epochs can lead to better performance but may cause overfitting.
    
    Returns:
    str: A message indicating the completion of training.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank: Higher values can capture more complex adaptations but require more memory
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,  # LoRA alpha: Scales the LoRA update, usually set to the same value as the LoRA rank
        lora_dropout=0,  # LoRA dropout: Can help prevent overfitting, 0 is optimized for speed
        bias="none",  # Bias configuration: "none" is optimized for speed
        use_gradient_checkpointing="unsloth",  # Gradient checkpointing: Reduces memory usage at the cost of increased computation time
        random_state=3407,  # Random seed for reproducibility
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Accumulate gradients over multiple steps to simulate larger batch sizes
            warmup_steps=5,  # Number of steps for learning rate warm-up
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),  # Use FP16 precision if bfloat16 is not supported
            bf16=is_bfloat16_supported(),  # Use bfloat16 precision if supported
            logging_steps=1,
            optim="adamw_8bit",  # 8-bit Adam optimizer for memory efficiency
            weight_decay=0.01,  # L2 regularization to prevent overfitting
            lr_scheduler_type="linear",  # Linear learning rate decay
            output_dir="outputs",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()
    return "Training completed successfully!"