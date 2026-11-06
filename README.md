# 🧠 AI & Machine Learning Knowledge Assistant

An intelligent document-based question-answering system powered by **Retrieval-Augmented Generation (RAG)**, Google Gemini, ChromaDB, and Sentence Transformers.

The application allows users to ask questions about Artificial Intelligence and Machine Learning documents and receive context-aware answers based on retrieved document content, with relevant source references and page citations.

---

## 📌 Project Overview

The AI & Machine Learning Knowledge Assistant combines semantic search with Large Language Models to provide document-grounded answers.

Instead of relying only on the knowledge stored within an LLM, this system retrieves relevant information from a custom knowledge base before passing the retrieved context to Google Gemini for answer generation.

The application also provides a comparison between traditional Gemini responses and RAG-based responses, helping users understand the benefits of document retrieval.

## ✨ Key Features

- **PDF and Markdown Ingestion:** Extract text from PDF and Markdown documents.
- **Intelligent Text Chunking:** Divide documents into smaller, overlapping text segments.
- **Semantic Search:** Find relevant document passages using Sentence Transformers.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **AI-Powered Answer Generation:** Generate context-aware responses using Google Gemini.
- **Source Citations:** Display document names and page numbers for retrieved information.
- **Multiple Answer Modes:**
  - RAG Answer – Generates answers using retrieved document context.
  - Plain Gemini – Generates answers without document retrieval.
  - Compare Both – Compares direct Gemini and RAG-generated answers.
- **Interactive Dashboard:** Explore the knowledge base through a Streamlit interface.
- **Persistent Storage:** Maintain document embeddings locally using ChromaDB.

## 🖥️ Application Preview

### Main Dashboard

<img width="1266" height="619" alt="image" src="https://github.com/user-attachments/assets/d21119bb-f7d2-4592-b8df-8b1f6161a164" />

<img width="1259" height="618" alt="image" src="https://github.com/user-attachments/assets/89a69e33-c272-468f-a336-df9d996f252b" />

<img width="1261" height="615" alt="image" src="https://github.com/user-attachments/assets/102fa025-7fec-4b1c-9ee4-835411bc2a24" />


### RAG Answer Generation

<img width="1263" height="627" alt="image" src="https://github.com/user-attachments/assets/bd7b99e3-c6f6-4ca4-a60e-43e6f56cee03" />
<img width="1253" height="604" alt="image" src="https://github.com/user-attachments/assets/5ac0ef1e-92b8-4325-825d-400a1689da1a" />

### No RAG Answer Generation

<img width="1262" height="621" alt="image" src="https://github.com/user-attachments/assets/a0b37204-296f-44b4-9cd5-2eb9fca9b360" />
<img width="1249" height="610" alt="image" src="https://github.com/user-attachments/assets/3ce53e52-c997-4456-8782-34f5bd80bab4" />


### RAG vs Plain Gemini Comparison

<img width="1255" height="610" alt="image" src="https://github.com/user-attachments/assets/48f71ba6-9538-43a9-8f7c-9892004c43ff" />
<img width="1249" height="615" alt="image" src="https://github.com/user-attachments/assets/74afb178-1cf1-45fa-8842-f46db1e4f3c7" />
<img width="1250" height="641" alt="image" src="https://github.com/user-attachments/assets/4bbd0f60-259d-4efe-8a34-3d99fe8d42ac" />

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit, Custom CSS |
| Large Language Model | Google Gemini |
| Embedding Model | Sentence Transformers |
| Embedding Model Name | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Text Splitting | LangChain Text Splitters |
| Environment Management | Python Virtual Environment |

## 🏗️ System Architecture

### RAG Pipeline

```text
                 USER
                   |
                   v
           Streamlit Dashboard
                   |
                   v
            Enter a Question
                   |
                   v
          Sentence Transformers
             Query Embedding
                   |
                   v
                ChromaDB
           Semantic Retrieval
                   |
                   v
         Retrieve Relevant Chunks
                   |
                   v
          Context Construction
                   |
                   v
             Google Gemini
                   |
                   v
          Context-Aware Answer
                   |
                   v
            Source Citations
                   |
                   v
            DISPLAY RESULT
```

### Document Ingestion Pipeline

```text
         PDF / Markdown Documents
                    |
                    v
             Text Extraction
                    |
                    v
              Text Chunking
                    |
                    v
          Sentence Transformers
            Embedding Generation
                    |
                    v
                ChromaDB
                    |
                    v
             Persistent Storage
```

## 📂 Project Structure

```text
RAG-Knowledge-Assistant/
│
├── app.py                  # Streamlit application
├── config.py               # Project configuration
├── ingest.py               # Document ingestion pipeline
├── retrieval.py            # Semantic search and retrieval
├── rag.py                  # RAG and Gemini integration
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
├── .env                    # API key configuration (not committed)
│
├── data/
│   ├── machine_learning_fundamentals.pdf
│   ├── supervised_learning.pdf
│   ├── neural_networks.pdf
│   ├── model_evaluation.pdf
│   └── rag_benchmark_note.md
│
└── chroma_db/              # Generated vector database (not committed)
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/VijayReddy-07/RAG-Knowledge-Assistant.git
```

```bash
cd RAG-Knowledge-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Generate an API key through [Google AI Studio](https://aistudio.google.com/apikey).

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
```

**Important:** Use a Gemini model currently supported by your API key. Never upload your actual API key to GitHub.

### 5. Add Knowledge Base Documents

Place your PDF and Markdown documents inside the `data/` directory.

Supported document formats in the current ingestion pipeline:

- PDF (`.pdf`)
- Markdown (`.md`)

### 6. Run Document Ingestion

Execute:

```bash
python ingest.py
```

The ingestion pipeline performs the following operations:

1. Reads PDF and Markdown documents.
2. Extracts text and document metadata.
3. Splits the extracted content into overlapping chunks.
4. Generates vector embeddings using Sentence Transformers.
5. Stores document chunks, embeddings, and metadata in ChromaDB.

### 7. Launch the Application

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the Knowledge Assistant.

## 💬 Sample Questions

Try asking the following questions:

1. What is Machine Learning?
2. Explain supervised learning.
3. What is the difference between classification and regression?
4. Explain the architecture of neural networks.
5. What is overfitting in Machine Learning?
6. Explain different model evaluation metrics.
7. What are the applications of deep learning?
8. Explain the working principles of a neural network.

## 🔍 RAG vs Plain Gemini

The application provides three answer modes.

| Mode | Description |
|---|---|
| RAG Answer | Retrieves relevant document passages and generates a context-grounded answer. |
| Plain Gemini | Generates an answer directly using Google Gemini without document retrieval. |
| Compare Both | Displays both approaches for comparison. |

### Why RAG?

Traditional LLM responses may not contain information from private or custom documents.

RAG addresses this limitation by retrieving relevant information from an external knowledge base and supplying it as context to the language model.

This helps improve document relevance, transparency, and source traceability.

## 📊 Knowledge Base

The knowledge base contains educational material covering:

- Machine Learning Fundamentals
- Supervised Learning
- Neural Networks
- Model Evaluation
- Custom RAG Benchmark Information

ChromaDB maintains the generated vector embeddings locally for semantic retrieval.

## 🔮 Future Enhancements

- Support for DOCX, TXT, and additional document formats.
- Upload documents directly through the dashboard.
- Multi-document knowledge-base management.
- Conversational chat history.
- Advanced retrieval and reranking techniques.
- Response evaluation and retrieval performance metrics.
- User authentication and document management.
- Cloud deployment and scalable vector storage.

## 🎯 Learning Outcomes

Through this project, I explored and implemented:

- Retrieval-Augmented Generation (RAG).
- Semantic search and vector embeddings.
- Document processing and text chunking.
- Vector database integration.
- Large Language Model API integration.
- Context-aware question answering.
- Building interactive applications using Streamlit.
- Comparing LLM-only and retrieval-grounded responses.

---

⭐ If you find this project useful, consider giving the repository a star!

**Built with Python | Google Gemini | ChromaDB | Sentence Transformers | Streamlit**
