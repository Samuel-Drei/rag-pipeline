# Simple RAG Pipeline

A Retrieval Augmented Generation (RAG) system that answers questions based on **your PDF documents**, with **Streamlit** web interface and **LanceDB** vector storage. Automatic deployment on Streamlit Cloud.


# RAG Pipeline v2.0 


| Metric | v1.0 | v2.0 | **Improvement** |
|--------|------|------|-----------------|
| **Accuracy** | 96% (24/25) | 88% (22/25) | **-8%** |
| **Avg Latency** | 15.8s | **5.2s** | **+67% faster** |
| **Max Latency** | 25s | 10.6s | **+58% faster** |
| **Token Cost** | 100% | **33%** | **+67% savings** |
| **Stability** (Std Dev) | 4.47s | 2.46s | **+45% more stable** |

**Net Win: 67% faster + 67% cheaper outweighs -8% accuracy!**


![rag-image](./rag-design-basic.png)

## Overview

The RAG Framework lets you:

- **Index Documents:** Process and break documents (e.g., PDFs) into smaller, manageable chunks.
- **Store & Retrieve Information:** Save document embeddings in a vector database (using LanceDB) and search using  similarity search + Cohere rerank.
- **Generate Responses:** Use an AI model (via the OpenAI API) to provide concise answers based on the retrieved context.
- **Evaluate Responses:** Compare the generated response against expected answers and view the reasoning behind the evaluation.
- **Web interface** → Streamlit (local + cloud)

## Architecture

- **Pipeline (src/rag_pipeline.py):**  
  Orchestrates the process using:

  - **Datastore:** Manages embeddings and vector storage.
  - **Indexer:** Processes documents and creates data chunks. Two versions are available—a basic PDF indexer and one using the Docling package.
  - **Retriever:** Searches the datastore to pull relevant document segments.
  - **ResponseGenerator:** Generates answers by calling the AI service.
  - **Evaluator:** Compares the AI responses to expected answers and explains the outcome.

- **Interfaces (interface/):**  
  Abstract base classes define contracts for all components (e.g., BaseDatastore, BaseIndexer, BaseRetriever, BaseResponseGenerator, and BaseEvaluator), making it easy to extend or swap implementations.

##  Quick Start

### 1. Clone & Setup
```
git clone https://github.com/Samuel-Drei/rag-pipeline.git
cd rag-pipeline
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Configure API Keys (.env)

text
OPENAI_API_KEY=sk-proj-your-key-here

CO_API_KEY=your-cohere-key-here

You will also need a Cohere key for the re-ranking feature used in `src/impl/retriever.py`. You can create an account and create an API key at https://cohere.com/

```sh
set -x CO_API_KEY "xxx"
```

## Usage

The CLI provides several commands to interact with the RAG pipeline. By default, they will use the source/eval paths specified in `main.py`, but there are flags to override them.

```python
DEFAULT_SOURCE_PATH = "sample_data/source/"
DEFAULT_EVAL_PATH = "sample_data/eval/sample_questions.json"
```

#### Run the Full Pipeline

This command resets the datastore, indexes documents, and evaluates the model.

```bash
python main.py run
```

#### Reset the Database

Clears the vector database.

```bash
python main.py reset
```

#### Add Documents

Index and embed documents. You can specify a file or directory path.

```bash
python main.py add -p "sample_data/source/"
```

#### Query the Database

Search for information using a query string.

```bash
python main.py query "What is the opening year of The Lagoon Breeze Hotel?"
```

#### Evaluate the Model

Use a JSON file (with question/answer pairs) to evaluate the response quality.

```bash
python main.py evaluate -f "sample_data/eval/sample_questions.json"

#### Run locally

streamlit run app.py

Opens localhost:8501


#### Streamlit Cloud Deployment

    Push to GitHub → auto-detected

    Streamlit Cloud → Settings → Secrets:


OPENAI_API_KEY = "sk-proj-..."
CO_API_KEY = "co-..."
```
