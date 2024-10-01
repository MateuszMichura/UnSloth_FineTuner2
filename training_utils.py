from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

def finetune_model(model, tokenizer, dataset, learning_rate, batch_size, num_epochs):
    """
    Fine-tune a model on a given dataset, using CUDA if available.
    This version supports fine-tuning of quantized models using PEFT and Unsloth optimizations.
    
    Args:
    model: The pre-trained model to fine-tune.
    tokenizer: The tokenizer associated with the model.
    dataset: The dataset to use for fine-tuning.
    learning_rate (float): Learning rate for optimization.
    batch_size (int): Number of training examples used in one iteration.
    num_epochs (int): Number of complete passes through the dataset.
    
    Returns:
    SFTTrainer: The trained model wrapped in an SFTTrainer object.
    """
    # Prepare the model for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Set up the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model.config.max_position_embeddings,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Apply train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Train the model
    trainer.train()

    return trainer