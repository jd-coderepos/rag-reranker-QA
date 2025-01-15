# 📑 RAG with Cross-Encoders Re-ranking Demo Application

This is a demo project showcasing a Large Language Model (LLM) integrated with Retrieval-Augmented Generation (RAG). The project demonstrates how to utilize LLMs for advanced document retrieval and text generation tasks.

## Requirements

- Python >= 3.10
- SQLite >= 3.35
- [Ollama](https://ollama.dev/download)

## Setting Up Locally

To set up this project on your local machine, follow these steps:

### 1. Install Python and SQLite

Make sure you have Python 3.10 or greater installed. You can download Python from the official [Python website](https://www.python.org/). Additionally, ensure you have SQLite version 3.35 or higher. SQLite is typically pre-installed with Python, but you can check your version by running:

```bash
sqlite3 --version
```

### 2. Install Dependencies

Make sure you have installed all these dependencies.

```sh
pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community
```

```sh
ollama pull llama3.2:3b
```

```sh
ollama run llama3.2
```

### 3. Run the Application

Run this application using the following command:

```sh
streamlit run app.py
```