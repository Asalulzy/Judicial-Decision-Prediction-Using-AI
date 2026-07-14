# Judicial Decision Prediction Using AI and Historical Court Records

## Overview

This project aims to predict prison sentence duration from Indonesian criminal court decisions using Natural Language Processing (NLP) and machine learning techniques. The system analyzes textual case descriptions, legal arguments, witness statements, and applied statutes to estimate sentencing outcomes based on patterns learned from historical judicial records.

The project was initially developed as a legal analytics solution using text embeddings and supervised machine learning models. To enhance explainability and legal information retrieval, a Retrieval-Augmented Generation (RAG) pipeline was later integrated, allowing users to explore similar past cases and supporting legal references alongside prediction results.

## Key Features

* Predict prison sentence duration from court decision documents.
* Process and analyze Indonesian legal text using NLP techniques.
* Generate semantic document representations using IndoBERT embeddings.
* Train predictive models using XGBoost on more than 23,600 court records.
* Retrieve relevant historical cases through a FAISS vector database.
* Provide explainable legal insights using a Retrieval-Augmented Generation (RAG) framework powered by LangChain and a local LLaMA-based Large Language Model (LLM).

## Technical Stack

* **Language:** Python
* **NLP:** IndoBERT
* **Machine Learning:** XGBoost
* **Vector Database:** FAISS
* **LLM Framework:** LangChain
* **Local LLM:** LLaMA-based model
* **Architecture:** Retrieval-Augmented Generation (RAG)

## Results

* Trained on **23,600+ Indonesian court decision documents**.
* Achieved approximately **82% prediction accuracy** for prison sentence duration.
* Improved legal document retrieval and contextual explanation through semantic search and RAG integration.

## Motivation

Indonesia's judiciary produces hundreds of thousands of court decisions every year. Identifying relevant precedents and understanding sentencing patterns can be time-consuming for legal practitioners and researchers. This project demonstrates how AI can assist legal analysis by combining predictive modeling with retrieval-based legal reasoning, enabling faster access to historical case information and more interpretable decision support.
