# LLM Finetuner

This project provides a user-friendly interface for fine-tuning Large Language Models (LLMs) using the Unsloth library. It includes features for dataset preparation, synthetic dataset creation, model training, testing, and GGUF conversion.

## Features

- Load and fine-tune various pre-trained models
- Prepare existing datasets or create synthetic datasets
- Fine-tune models with customizable hyperparameters
- Test fine-tuned models
- Convert models to GGUF format for deployment

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (for efficient training)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-finetuner.git
   cd llm-finetuner
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. Open the provided URL in your web browser to access the Gradio interface.

3. Follow these steps in the interface:
   a. Settings: Enter your Hugging Face token and select a model.
   b. Dataset: Prepare an existing dataset or create a synthetic one.
   c. Training: Set hyperparameters and start the fine-tuning process.
   d. Test: Test your fine-tuned model with custom inputs.
   e. GGUF Conversion: Convert your model to GGUF format if needed.

## Notes

- Ensure you have the necessary API keys for OpenAI or Anthropic if you plan to use them for synthetic dataset creation.
- If using Ollama for local generation, make sure it's installed and running on your machine.
- Fine-tuning can be computationally intensive. Ensure you have adequate GPU resources available.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.