# Project Overview

This project, "Ragger", is a Retrieval-Augmented Generation (RAG) style AI management system. It's designed to process various document types, store them in a vector database, and use them to answer user queries.

The project is split into two main components:

*   `ragger_data.py`: A data processing application that handles document ingestion, parsing, chunking, embedding generation, and storage. It supports PDF, TXT, Markdown, and DOCX files. It uses a SQLite database to track processed files and Weaviate as the vector store.
*   `ragger.py`: An AI interaction application that takes user queries, retrieves relevant context from the vector database, and generates a response.

## Technologies Used

*   **Programming Language:** Python
*   **Vector Database:** Weaviate
*   **Embedding Model:** SentenceTransformer (e.g., `all-MiniLM-L6-v2`)
*   **Data Handling:** SQLite, pdfplumber, python-docx, Markdown, PyPDF2
*   **Core Libraries:** `weaviate-client`, `sentence-transformers`, `torch`

## Building and Running

### 1. Installation

First, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Data Processor

To process and ingest documents, use `ragger_data.py`. You'll need a running Weaviate instance.

**Process a single file:**

```bash
python ragger_data.py --process-file /path/to/your/document.pdf
```

**Process a directory:**

```bash
python ragger_data.py --process-dir /path/to/your/documents --recursive
```

**List processed documents:**

```bash
python ragger_data.py --list-docs
```

### 3. Running the AI Interaction

To interact with the RAG system, use `ragger.py`.

**Run in interactive mode:**

```bash
python ragger.py --interactive
```

**Run with a single query:**

```bash
python ragger.py --query "Your question here"
```

## Development Conventions

*   The project follows a standard Python project structure.
*   Configuration is managed through a `ragger_config.json` file. A default configuration is provided if the file doesn't exist.
*   Logging is configured to output to both the console and log files (`ragger_ai.log`, `ragger_data.log`).
*   The code is modular, with clear separation of concerns between data processing and AI interaction.
*   Command-line arguments are used to control the behavior of the scripts.
