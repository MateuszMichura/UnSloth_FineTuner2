import gradio as gr
from model_utils import load_model
from dataset_utils import prepare_dataset, create_synthetic_dataset
from training_utils import finetune_model
from inference_utils import test_model
from gguf_utils import convert_to_gguf
from upload_utils import upload_to_huggingface

def create_gradio_interface():
    models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",
        "unsloth/Llama-3.2-1B-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct",
    ]

    with gr.Blocks() as demo:
        gr.Markdown("# LLM Finetuner")
        
        model = gr.State(None)
        tokenizer = gr.State(None)
        dataset = gr.State(None)
        
        with gr.Tab("Settings"):
            hf_token = gr.Textbox(label="Hugging Face Token", type="password")
            model_path = gr.Dropdown(label="Model", choices=models, value="unsloth/Llama-3.2-3B-Instruct")
            load_model_btn = gr.Button("Load Model")
            load_model_output = gr.Textbox(label="Load Model Output")
        
        with gr.Tab("Dataset"):
            with gr.Group():
                gr.Markdown("## Use Existing Dataset")
                dataset_source = gr.Radio(["Hugging Face", "Local File"], label="Dataset Source", value="Hugging Face")
                hf_dataset_path = gr.Textbox(label="Hugging Face Dataset Path", value="mlabonne/FineTome-100k")
                local_dataset_path = gr.File(label="Upload Local Dataset (JSON or CSV)", visible=False)
                prepare_dataset_btn = gr.Button("Prepare Dataset")
                prepare_dataset_output = gr.Textbox(label="Prepare Dataset Output")
            
            with gr.Group():
                gr.Markdown("## Create Synthetic Dataset")
                examples = gr.Textbox(label="Example Conversations", lines=10, placeholder="Enter example conversations here...")
                expected_structure = gr.Textbox(label="Expected Dataset Structure", lines=5, placeholder="Enter the expected structure for the dataset...")
                num_samples = gr.Number(label="Number of Samples to Generate", value=100)
                ai_provider = gr.Radio(["OpenAI", "Anthropic", "Ollama"], label="AI Provider")
                api_key = gr.Textbox(label="API Key", type="password")
                ollama_model = gr.Textbox(label="Ollama Model Name", visible=False)
                create_dataset_btn = gr.Button("Create Synthetic Dataset")
                create_dataset_output = gr.Textbox(label="Create Dataset Output")
        
        with gr.Tab("Training"):
            gr.Markdown("## Hyperparameters")
            with gr.Row():
                learning_rate = gr.Number(label="Learning Rate", value=2e-4)
                lr_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                batch_size = gr.Number(label="Batch Size", value=2)
                bs_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                num_epochs = gr.Number(label="Number of Epochs", value=1)
                epochs_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                weight_decay = gr.Number(label="Weight Decay", value=0.01)
                wd_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                warmup_steps = gr.Number(label="Warmup Steps", value=0)
                warmup_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=1)
                gas_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                max_seq_length = gr.Number(label="Max Sequence Length", value=512)
                msl_info = gr.Button("ℹ️", elem_classes="info-button")
            with gr.Row():
                packing = gr.Checkbox(label="Enable Prompt Packing", value=True)
                packing_info = gr.Button("ℹ️", elem_classes="info-button")
            
            train_btn = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output")

            # Info markdown components
            lr_info_md = gr.Markdown(visible=False)
            bs_info_md = gr.Markdown(visible=False)
            epochs_info_md = gr.Markdown(visible=False)
            wd_info_md = gr.Markdown(visible=False)
            warmup_info_md = gr.Markdown(visible=False)
            gas_info_md = gr.Markdown(visible=False)
            msl_info_md = gr.Markdown(visible=False)
            packing_info_md = gr.Markdown(visible=False)
        
        with gr.Tab("Test"):
            test_input = gr.Textbox(label="Test Input")
            test_btn = gr.Button("Test Model")
            test_output = gr.Textbox(label="Model Output")

        with gr.Tab("GGUF Conversion"):
            gguf_output_path = gr.Textbox(label="GGUF Output Path")
            gguf_quant_method = gr.Dropdown(
                label="Quantization Method",
                choices=["q8_0", "q4_k_m", "q5_k_m", "f16"],
                value="q8_0"
            )
            gguf_convert_btn = gr.Button("Convert to GGUF")
            gguf_output = gr.Textbox(label="GGUF Conversion Output")

        with gr.Tab("Upload to Hugging Face"):
            repo_name = gr.Textbox(label="Repository Name")
            upload_btn = gr.Button("Upload to Hugging Face")
            upload_output = gr.Textbox(label="Upload Output")

        def load_model_and_tokenizer(model_path, hf_token):
            model_val, tokenizer_val = load_model(model_path, hf_token)
            return model_val, tokenizer_val, "Model and tokenizer loaded successfully!"

        def update_ollama_visibility(choice):
            return gr.update(visible=(choice == "Ollama"))

        def update_dataset_input_visibility(choice):
            return gr.update(visible=(choice == "Hugging Face")), gr.update(visible=(choice == "Local File"))

        load_model_btn.click(
            load_model_and_tokenizer,
            inputs=[model_path, hf_token],
            outputs=[model, tokenizer, load_model_output]
        )

        dataset_source.change(
            update_dataset_input_visibility,
            inputs=[dataset_source],
            outputs=[hf_dataset_path, local_dataset_path]
        )

        def prepare_dataset_wrapper(source, hf_path, local_file, hf_token, tokenizer_val):
            if tokenizer_val is None:
                return "Error: Model and tokenizer not loaded. Please load the model first."
            
            if source == "Hugging Face":
                dataset_val = prepare_dataset("huggingface", hf_path, tokenizer_val, hf_token)
            elif source == "Local File":
                if local_file is not None:
                    dataset_val = prepare_dataset("local", local_file.name, tokenizer_val)
                else:
                    return "No file uploaded. Please upload a local dataset file."
            else:
                return "Invalid dataset source selected."
            
            return dataset_val, "Dataset prepared successfully!"

        prepare_dataset_btn.click(
            prepare_dataset_wrapper,
            inputs=[dataset_source, hf_dataset_path, local_dataset_path, hf_token, tokenizer],
            outputs=[dataset, prepare_dataset_output]
        )
        
        def create_synthetic_dataset_wrapper(examples, expected_structure, num_samples, ai_provider, api_key, ollama_model, tokenizer_val):
            if tokenizer_val is None:
                return "Error: Model and tokenizer not loaded. Please load the model first."
            
            dataset_val = create_synthetic_dataset(examples, expected_structure, num_samples, ai_provider, api_key, ollama_model)
            return dataset_val, "Synthetic dataset created successfully!"

        create_dataset_btn.click(
            create_synthetic_dataset_wrapper,
            inputs=[examples, expected_structure, num_samples, ai_provider, api_key, ollama_model, tokenizer],
            outputs=[dataset, create_dataset_output]
        )
        
        ai_provider.change(update_ollama_visibility, inputs=[ai_provider], outputs=[ollama_model])
        
        def train_model_wrapper(model_val, tokenizer_val, dataset_val, learning_rate, batch_size, num_epochs, weight_decay, warmup_steps, gradient_accumulation_steps, max_seq_length, packing):
            if model_val is None or tokenizer_val is None:
                return "Error: Model and tokenizer not loaded. Please load the model first."
            if dataset_val is None:
                return "Error: Dataset not prepared. Please prepare or create a dataset first."
            
            try:
                output = finetune_model(model_val, tokenizer_val, dataset_val, learning_rate, batch_size, num_epochs, weight_decay, warmup_steps, gradient_accumulation_steps, max_seq_length, packing)
                return output
            except Exception as e:
                return f"Error during training: {str(e)}"

        train_btn.click(
            train_model_wrapper,
            inputs=[model, tokenizer, dataset, learning_rate, batch_size, num_epochs, weight_decay, warmup_steps, gradient_accumulation_steps, max_seq_length, packing],
            outputs=[train_output]
        )
        
        def test_model_wrapper(model_val, tokenizer_val, test_input):
            if model_val is None or tokenizer_val is None:
                return "Error: Model and tokenizer not loaded. Please load the model first."
            
            output = test_model(model_val, tokenizer_val, test_input)
            return output

        test_btn.click(
            test_model_wrapper,
            inputs=[model, tokenizer, test_input],
            outputs=[test_output]
        )

        def convert_to_gguf_wrapper(model_val, tokenizer_val, gguf_output_path, gguf_quant_method):
            if model_val is None or tokenizer_val is None:
                return "Error: Model and tokenizer not loaded. Please load the model first."
            
            output = convert_to_gguf(model_val, tokenizer_val, gguf_output_path, gguf_quant_method)
            return output

        gguf_convert_btn.click(
            convert_to_gguf_wrapper,
            inputs=[model, tokenizer, gguf_output_path, gguf_quant_method],
            outputs=[gguf_output]
        )

        def upload_model_wrapper(model_val, tokenizer_val, repo_name, hf_token):
            if model_val is None or tokenizer_val is None:
                return "Error: Model and tokenizer not loaded. Please load the model first."
            
            output = upload_to_huggingface(model_val, tokenizer_val, repo_name, hf_token)
            return output

        upload_btn.click(
            upload_model_wrapper,
            inputs=[model, tokenizer, repo_name, hf_token],
            outputs=[upload_output]
        )

        # Info button click events
        lr_info.click(lambda: gr.update(visible=True, value="""
        **Learning Rate**
        - Controls the step size during optimization.
        - Typical values: 1e-5 to 1e-3
        - Lower values: slower learning, more stable
        - Higher values: faster learning, risk of instability
        - Start with 2e-4 and adjust based on performance
        """), outputs=lr_info_md)

        bs_info.click(lambda: gr.update(visible=True, value="""
        **Batch Size**
        - Number of samples processed before the model is updated.
        - Larger batch sizes can lead to faster training but require more memory.
        - Start with 2-4 for most GPUs, increase if you have more GPU memory.
        - If you encounter out-of-memory errors, reduce this value.
        """), outputs=bs_info_md)

        epochs_info.click(lambda: gr.update(visible=True, value="""
        **Number of Epochs**
        - Number of complete passes through the training dataset.
        - More epochs can lead to better performance but risk overfitting.
        - Start with 1-3 epochs and increase if the model is underfitting.
        - Monitor validation loss to prevent overfitting.
        """), outputs=epochs_info_md)

        wd_info.click(lambda: gr.update(visible=True, value="""
        **Weight Decay**
        - L2 regularization term to prevent overfitting.
        - Typical values: 0.01 to 0.1
        - Higher values result in stronger regularization.
        - If your model is overfitting, try increasing this value.
        """), outputs=wd_info_md)

        warmup_info.click(lambda: gr.update(visible=True, value="""
        **Warmup Steps**
        - Number of steps for learning rate warmup.
        - Gradually increases learning rate from 0 to the set value.
        - Can help stabilize training in the beginning.
        - Try 10% of total training steps or 100-500 steps.
        """), outputs=warmup_info_md)

        gas_info.click(lambda: gr.update(visible=True, value="""
        **Gradient Accumulation Steps**
        - Number of steps to accumulate gradients before updating.
        - Allows for larger effective batch sizes with limited memory.
        - If set to N, the effective batch size is N * batch_size.
        - Increase this if you want a larger batch size but face memory issues.
        """), outputs=gas_info_md)

        msl_info.click(lambda: gr.update(visible=True, value="""
        **Max Sequence Length**
        - Maximum length of sequences after tokenization.
        - Longer sequences will be truncated, shorter ones padded.
        - Adjust based on your model's architecture and your data.
        - Typical values: 512, 1024, 2048 (model-dependent)
        """), outputs=msl_info_md)

        packing_info.click(lambda: gr.update(visible=True, value="""
        **Enable Prompt Packing**
        - Efficiently packs multiple short prompts into a single sequence.
        - Can significantly speed up training for short sequences.
        - Recommended for most fine-tuning tasks.
        - Disable if you encounter issues or for very long sequences.
        """), outputs=packing_info_md)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()