import gradio as gr
from model_utils import load_model
from dataset_utils import prepare_dataset, create_synthetic_dataset
from training_utils import finetune_model
from inference_utils import test_model
from gguf_utils import convert_to_gguf

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
        "unsloth/Llama-3.2-3B-Instruct"
    ]

    with gr.Blocks() as demo:
        gr.Markdown("# LLM Finetuner")
        
        with gr.Tab("Settings"):
            hf_token = gr.Textbox(label="Hugging Face Token", type="password")
            model_path = gr.Dropdown(label="Model", choices=models, value="unsloth/Llama-3.2-3B-Instruct")
        
        with gr.Tab("Dataset"):
            with gr.Group():
                gr.Markdown("## Use Existing Dataset")
                dataset_source = gr.Radio(["Hugging Face", "Local File"], label="Dataset Source", value="Hugging Face")
                hf_dataset_path = gr.Textbox(label="Hugging Face Dataset Path", value="mlabonne/FineTome-100k")
                local_dataset_path = gr.File(label="Upload Local Dataset (JSON or CSV)", visible=False)
                convert_btn = gr.Button("Prepare Dataset")
            
            with gr.Group():
                gr.Markdown("## Create Synthetic Dataset")
                examples = gr.Textbox(label="Example Conversations", lines=10, placeholder="Enter example conversations here...")
                expected_structure = gr.Textbox(label="Expected Dataset Structure", lines=5, placeholder="Enter the expected structure for the dataset...")
                num_samples = gr.Number(label="Number of Samples to Generate", value=100)
                ai_provider = gr.Radio(["OpenAI", "Anthropic", "Ollama"], label="AI Provider")
                api_key = gr.Textbox(label="API Key", type="password")
                ollama_model = gr.Textbox(label="Ollama Model Name", visible=False)
                create_dataset_btn = gr.Button("Create Synthetic Dataset")
        
        with gr.Tab("Training"):
            learning_rate = gr.Number(label="Learning Rate", value=2e-4)
            batch_size = gr.Number(label="Batch Size", value=2)
            num_epochs = gr.Number(label="Number of Epochs", value=1)
            train_btn = gr.Button("Start Training")
            train_output = gr.Textbox(label="Training Output")
        
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

        model = gr.State()
        tokenizer = gr.State()
        dataset = gr.State()

        def load_model_and_tokenizer(model_path, hf_token):
            model, tokenizer = load_model(model_path, hf_token)
            return model, tokenizer, "Model loaded successfully!"

        def update_ollama_visibility(choice):
            return gr.update(visible=(choice == "Ollama"))

        def update_dataset_input_visibility(choice):
            return gr.update(visible=(choice == "Hugging Face")), gr.update(visible=(choice == "Local File"))

        load_model_btn = gr.Button("Load Model")
        load_model_btn.click(load_model_and_tokenizer, inputs=[model_path, hf_token], outputs=[model, tokenizer, train_output])

        dataset_source.change(update_dataset_input_visibility, inputs=[dataset_source], outputs=[hf_dataset_path, local_dataset_path])

        def prepare_dataset_wrapper(source, hf_path, local_file, hf_token):
            if source == "Hugging Face":
                return prepare_dataset("huggingface", hf_path, tokenizer.value, hf_token)
            elif source == "Local File":
                if local_file is not None:
                    return prepare_dataset("local", local_file.name, tokenizer.value)
                else:
                    return "No file uploaded. Please upload a local dataset file."
            else:
                return "Invalid dataset source selected."

        convert_btn.click(prepare_dataset_wrapper, 
                          inputs=[dataset_source, hf_dataset_path, local_dataset_path, hf_token], 
                          outputs=[dataset])
        
        def create_synthetic_dataset_wrapper(examples, expected_structure, num_samples, ai_provider, api_key, ollama_model):
            dataset = create_synthetic_dataset(examples, expected_structure, num_samples, ai_provider, api_key, ollama_model)
            return dataset, "Synthetic dataset created successfully!"

        create_dataset_btn.click(create_synthetic_dataset_wrapper, 
                                 inputs=[examples, expected_structure, num_samples, ai_provider, api_key, ollama_model], 
                                 outputs=[dataset, train_output])
        
        ai_provider.change(update_ollama_visibility, inputs=[ai_provider], outputs=[ollama_model])
        
        train_btn.click(finetune_model, inputs=[model, tokenizer, dataset, learning_rate, batch_size, num_epochs], outputs=[train_output])
        
        test_btn.click(test_model, inputs=[model, tokenizer, test_input], outputs=[test_output])

        gguf_convert_btn.click(convert_to_gguf, inputs=[model, tokenizer, gguf_output_path, gguf_quant_method], outputs=[gguf_output])

    return demo