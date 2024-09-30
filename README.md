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


# Comprehensive Python Setup Guide

This guide will walk you through setting up Python, creating a virtual environment, and running your LLM Finetuner project on a new system.

## 1. Install Python

### Windows:
1. Go to https://www.python.org/downloads/windows/
2. Download the latest Python 3.x installer (64-bit version recommended)
3. Run the installer
4. Check "Add Python to PATH" during installation
5. Click "Install Now"

### macOS:
1. Install Homebrew if you haven't already:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python using Homebrew:
   ```
   brew install python
   ```

### Linux (Ubuntu/Debian):
1. Update package list:
   ```
   sudo apt update
   ```
2. Install Python:
   ```
   sudo apt install python3 python3-pip python3-venv
   ```

## 2. Verify Python Installation

Open a terminal (Command Prompt on Windows) and run:
```
python --version
```
You should see the Python version number. If not, try `python3 --version`.

## 3. Install Git

### Windows:
1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. Use the default settings during installation

### macOS:
If you installed Homebrew earlier:
```
brew install git
```

### Linux (Ubuntu/Debian):
```
sudo apt install git
```

## 4. Clone the Repository

1. Open a terminal
2. Navigate to where you want to store the project
3. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-finetuner.git
   cd llm-finetuner
   ```

## 5. Create and Activate a Virtual Environment

### Windows:
```
python -m venv venv
venv\Scripts\activate
```

### macOS and Linux:
```
python3 -m venv venv
source venv/bin/activate
```

Your prompt should change to indicate that the virtual environment is active.

## 6. Install Required Packages

With the virtual environment activated:
```
pip install -r requirements.txt
```

This may take a while as it installs all necessary dependencies.

## 7. Set Up CUDA (for GPU support)

If you have an NVIDIA GPU and want to use it for training:

1. Go to https://developer.nvidia.com/cuda-downloads
2. Download and install the CUDA Toolkit appropriate for your system
3. Install the cuDNN library:
   - Go to https://developer.nvidia.com/cudnn
   - Download cuDNN (you may need to create an NVIDIA account)
   - Follow the installation instructions for your system

## 8. Run the Application

With the virtual environment still activated:
```
python main.py
```

This will start the Gradio interface. Open the provided URL in your web browser.

## 9. Using the LLM Finetuner

1. In the "Settings" tab:
   - Enter your Hugging Face token
   - Select a model

2. In the "Dataset" tab:
   - Prepare an existing dataset or create a synthetic one

3. In the "Training" tab:
   - Set hyperparameters and start training

4. In the "Test" tab:
   - Test your fine-tuned model

5. In the "GGUF Conversion" tab:
   - Convert your model to GGUF format if needed

## Troubleshooting

- If `python` doesn't work, try `python3`
- Ensure your GPU drivers are up to date for CUDA support
- If you encounter "command not found" errors, ensure the relevant programs are in your system's PATH

## Closing Notes

- Always activate the virtual environment before running the project
- To deactivate the virtual environment, simply type `deactivate` in the terminal
- Keep your Python packages updated with `pip install --upgrade -r requirements.txt`

Remember to keep your API keys and tokens secure. Happy fine-tuning!