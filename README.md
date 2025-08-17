
# Vale Rule Generator

A web-based tool to generate [Vale](https://vale.sh) linter rules using natural language. It uses AI models combined with the official Vale documentation to create accurate YAML configurations.

This tool is designed to lower the barrier to entry for creating custom Vale rules by translating plain English requests into valid rule syntax.

## Features

*   **Multiple AI providers**: Support for Google Gemini, OpenAI GPT, and Anthropic Claude models
*   **Rule alternatives with confidence scores**: Get 3 different rule approaches with confidence ratings (recommended and alternative options)
*   **Smart rule generation**: Uses retrieval-augmented generation (RAG) with Vale documentation and real rule examples
*   **Professional examples**: Pre-loaded prompts from Google, Microsoft, and Splunk style guides
*   **Interactive UI**: Clean web interface with collapsible alternatives, syntax highlighting, and copy buttons
*   **Progressive feedback**: Real-time status updates during rule generation
*   **Debug transparency**: View the exact documentation snippets used to generate rules

## How it works

This tool uses a Retrieval-Augmented Generation (RAG) pipeline. When you enter a request, the application searches a vector database containing the official Vale documentation and real rule examples from Vale core, Google, and Microsoft style guides. The most relevant documents are retrieved and provided to your chosen AI model as context, along with your request. This allows the model to generate multiple rule alternatives that are grounded in Vale's actual syntax and proven patterns.

## Requirements

*   Python 3.8+
*   At least one API key from:
    *   Google Gemini API
    *   OpenAI API
    *   Anthropic Claude API

## Setup and usage

Follow these steps to run the application locally.

### 1. Clone the repository

```bash
git clone <repository_url> # Replace with the actual URL
cd vale-rule-generator
```

### 2. Create a virtual environment (recommended)

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Copy the example `.env` file to `.env` and add your API keys.

```bash
cp .env.example .env
```

Open the `.env` file and add at least one API key:

```bash
GEMINI_API_KEY="your_gemini_key_here"
OPENAI_API_KEY="your_openai_key_here"  
ANTHROPIC_API_KEY="your_claude_key_here"
```

The application will automatically detect which providers are available based on your configured keys.

### 5. Run the one-time setup script

This script downloads the Vale documentation and builds the local vector database. This may take a few minutes.

```bash
python setup.py
```

### 6. Run the application

```bash
python app.py
```

The application will now be running at `http://127.0.0.1:5001`.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
