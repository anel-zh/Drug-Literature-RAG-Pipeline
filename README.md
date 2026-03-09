# Regulatory-Grade Hybrid RAG Pipeline for Drug Literature

## Introduction

Bioengineering and pharmacological datasets often consist of dense, unstructured PDF documents—such as FDA labels and clinical research papers—where precision is critical. Extracting reliable information from these sources requires more than simple keyword matching; it requires a system capable of capturing technical context while providing verifiable citations to reduce the risk of LLM hallucinations.

This repository presents a modular Python-based Hybrid RAG (Retrieval-Augmented Generation) pipeline. The workflow standardizes the transition from raw PDFs to a local searchable index, combining semantic vector retrieval, keyword-based search, and cross-encoder reranking to improve retrieval quality.

## Project Motivation

This case study explores the performance differences between **Vanilla RAG** (vector-only retrieval) and a **hybrid retrieval pipeline**. Technical biomedical documents frequently contain precise identifiers (e.g., drug names, regulatory codes, or trial IDs) that semantic embeddings may not always capture reliably, while keyword-based retrieval methods can preserve these exact matches.

To address this, the pipeline implements a dual-stream retrieval architecture:

- Dense Retrieval: Uses FAISS and embedding models to capture semantic similarity and conceptual queries.
- Sparse Retrieval: Uses BM25 scoring to preserve exact matching for technical terminology and drug identifiers.
- Re-ranking & Citations: Applies Reciprocal Rank Fusion (RRF) followed by a Cross-Encoder reranking stage to prioritize the most relevant passages and require page-level citations in generated responses.

## Methods

**Indexing Workflow:**  
The `01_build_index.py` module parses PDF documents, applies recursive character chunking, and builds a dual-store index (FAISS for embeddings and BM25 for keyword search). Document metadata (e.g., FDA labels vs. clinical papers) is inferred automatically to enable metadata-based routing.

**Hybrid Retrieval Pipeline:**  
The `02_retrieve.py` script executes the retrieval and generation process. A query router narrows the search space, a hybrid retriever performs dense and sparse search, and a Cross-Encoder reranker filters results before the final context is sent to a local LLM (e.g., Llama 3 or Mistral via Ollama).

**Deterministic Benchmarking:**  
The `03_benchmark.py` module evaluates system performance by comparing a baseline vector-only RAG system with the hybrid retrieval pipeline. Evaluation metrics include citation presence, information completeness, and response latency.

## Expected Results

The pipeline generates diagnostic artifacts and evaluation summaries, including:

- Retrieval Comparison Reports: Summaries comparing hybrid retrieval against dense-only retrieval.
- Citation Verification: Validation that generated responses reference specific document IDs and page numbers (e.g., `[HADLIMA_FDA p.12]`).
- Routing Diagnostics: Logs demonstrating the system’s ability to identify whether a query should target a specific regulatory document or broader clinical literature.

## Technical Features

This case study explores several retrieval and system design concepts implemented within the `src/` directory:

- Reciprocal Rank Fusion (RRF): A method for combining results from multiple search strategies without requiring score normalization.
- Local LLM Integration: Designed to run entirely on local hardware using quantized models, improving privacy and reducing operational cost.
- Metadata-Aware Routing: Queries are routed to relevant document subsets before retrieval, improving search precision and efficiency.

## Repository Note

This project is a technical case study focused on navigation and retrieval within pharmacological literature. It serves as a modular template for building high-precision RAG systems for scientific documents, with particular emphasis on reproducibility, citation grounding, and flexible system components.
