def test_model(model, tokenizer, input_text):
    """
    Test the fine-tuned model with a given input.
    
    Args:
    model: The fine-tuned model to test.
    tokenizer: The tokenizer associated with the model.
    input_text (str): The input text to generate a response for.
    
    Returns:
    str: The generated response from the model.
    """
    messages = [
        {"role": "user", "content": input_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=64000, use_cache=True,
                             temperature=0, min_p=0.1)
    return tokenizer.batch_decode(outputs)[0]
