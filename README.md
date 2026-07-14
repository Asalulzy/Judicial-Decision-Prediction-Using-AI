# ⚖️ Legal AI Assistant for Indonesian Court Decisions using RAG

An AI-powered legal document analysis system built using **Retrieval-Augmented Generation (RAG)** to understand and answer questions from Indonesian court decision documents.

This project enables users to explore legal cases, retrieve relevant historical court decisions, and generate context-aware answers based on retrieved legal documents using semantic search and Large Language Models (LLMs).

The system combines document processing, embedding-based retrieval, vector databases, and LLM reasoning to assist legal document analysis.

---

## 📌 Project Overview

Legal documents, especially court decisions, often contain hundreds of pages of complex information including:

- Case background
- Chronology of events
- Legal articles applied
- Prosecutor's demands
- Judge considerations
- Final verdict

Analyzing these documents manually can be time-consuming.

This project develops an AI assistant that can process Indonesian court decisions and answer legal questions by retrieving relevant information from historical legal documents and generating responses grounded in the retrieved context.

---

## ✨ Key Features

### 📄 Legal Document Processing

The system provides a pipeline to process raw court decision documents:

- Load legal documents from text files
- Clean and preprocess legal text
- Split long documents into smaller chunks
- Prepare documents for semantic retrieval

### 🔎 Semantic Legal Search

Instead of relying only on keyword matching, the system uses embedding-based search:

- Generate semantic representations of legal documents
- Store embeddings in a FAISS vector database
- Retrieve the most relevant legal passages based on user queries

### 🤖 Retrieval-Augmented Generation (RAG)

The core system follows a RAG architecture:

1. User submits a legal question
2. Retriever searches relevant document chunks
3. Retrieved context is provided to the LLM
4. LLM generates an answer based on retrieved legal information

This approach helps reduce hallucination by grounding responses in actual legal documents.

### 📑 Legal Information Extraction

The system can extract structured legal information, including:

- Case background
- Applied legal articles
- Charges
- Judge considerations
- Final decision

### 🖥️ Interactive User Interface

A Streamlit-based interface is provided for:

- Asking legal questions
- Searching relevant court decisions
- Viewing AI-generated explanations

---

## 🏗️ System Architecture

```
                 User Query
                      |
                      v
             Streamlit Interface
                      |
                      v
               RAG Pipeline
                      |
      --------------------------------
      |                              |
      v                              v
 Retriever                         LLM
      |
      v
FAISS Vector Database
      |
      v
Indonesian Court Documents
```

Detailed pipeline:

```
Court Documents (.txt)
        |
        v
   Document Loader
        |
        v
  Text Preprocessing
        |
        v
  Document Chunking
        |
        v
 Embedding Generation
        |
        v
FAISS Vector Database
        |
        v
 Semantic Retrieval
        |
        v
   LLM Generation
        |
        v
Legal Answer + Information Extraction
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| NLP Processing | Text Embedding Model |
| LLM Framework | LangChain |
| Vector Database | FAISS |
| Application Framework | Streamlit |
| Architecture | Retrieval-Augmented Generation (RAG) |
| Environment Management | Python Virtual Environment |

---

## 📂 Project Structure

```
Legal-AI-RAG/
│
├── src/
│   │
│   ├── app.py                  # Streamlit application
│   ├── rag_pipeline.py         # Main RAG orchestration pipeline
│   ├── data_loader.py          # Document loading module
│   ├── preprocessing.py        # Text cleaning and chunking
│   ├── embedding.py            # Embedding generation
│   ├── vector_database.py      # FAISS vector database management
│   ├── retriever.py            # Semantic document retrieval
│   ├── llm.py                  # LLM integration
│   ├── extractor.py            # Legal information extraction
│   │
│   ├── run_smoke_test.py       # Pipeline testing
│   └── run_ui_smoke.py         # UI testing
│
├── dataset/
│   └── README.md               # Dataset preparation guide
│
├── vectorstore/                # Local vector database storage
│
├── requirements.txt
├── .env.example
├── README.md
└── PROJECT_SUMMARY.txt
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Asalulzy/Judicial-Decision-Prediction-Using-AI.git
cd Legal-AI-RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy:

```bash
.env.example
```

to:

```bash
.env
```

Configure required variables such as:

```
DATASET_PATH=
MODEL_NAME=
API_KEY=
```

---

## 📚 Dataset Preparation

The original court decision dataset is not included in this repository due to size and privacy considerations.

Prepare your dataset structure:

```
dataset/
└── file_putusan/
    ├── doc_1.txt
    ├── doc_2.txt
    └── ...
```

Then configure the dataset path inside `.env`.

---

## ▶️ Running Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

The application will provide an interface for querying legal documents.

---

## 💡 Example Questions

The assistant can answer questions such as:

```
Apa kronologi kasus dalam putusan ini?
Pasal apa yang dikenakan kepada terdakwa?
Apa pertimbangan hakim dalam kasus tersebut?
Bagaimana hasil akhir putusan?
```

---

## 🧪 Testing

Run pipeline smoke test:

```bash
python src/run_smoke_test.py
```

Run UI smoke test:

```bash
python src/run_ui_smoke.py
```

---

## 🚀 Future Improvements

Future development plans:

- Hybrid search using BM25 + vector retrieval
- Legal document reranking model
- Better citation and source attribution
- Evaluation benchmark for RAG quality
- Fine-tuning Indonesian legal language models
- FastAPI backend deployment
- Docker containerization
- Cloud deployment and monitoring

---

## ⚠️ Disclaimer

This system is designed as an AI-assisted legal research tool.

The generated responses should not be considered as official legal advice or a replacement for professional legal judgment.

---
