# Document-Trained Customer Support Bot

## Overview
This project implements a customer support bot that trains on a provided document (FAQ, product manual, or policy file), answers customer queries based on that document, and iteratively improves its responses using simulated feedback.  

The bot demonstrates agentic workflow design, retrieval-augmented generation (RAG), and autonomous decision-making, aligning with modern AI/ML practices for production-ready systems.

## Features
- Document ingestion: Supports `.txt` and `.pdf` files (via PyPDF2).
- Semantic retrieval: Uses sentence-transformers embeddings with keyword fallback for robust section matching.
- Question answering: Hugging Face QA pipeline extracts answers from relevant context.
- Feedback loop: Simulates feedback (`good`, `too vague`, `not helpful`) and adjusts responses up to 2 iterations.
- Logging: Transparent logs of queries, feedback, and adjustments (`support_bot_log.txt`).
- Graceful fallback: Handles out-of-scope queries with clear responses.

## Installation

### Prerequisites
- Python 3.10
- Conda or venv recommended

### Setup
```bash
# Create environment
conda create --name supportbot python=3.10 -y
conda activate supportbot

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```text
torch==2.2.2
transformers==4.41.2
sentence-transformers==2.7.0
PyPDF2==3.0.1
```
TensorFlow/Keras is optional and not required for PyTorch-only mode.

## Usage

### Run with default queries
```bash
python support_bot_agent.py --document faq.txt
```

### Run with custom queries
```bash
python support_bot_agent.py --document faq.txt --queries "What's the refund policy?" "How do I contact support?"
```

### Example Output
```
Initial Response to 'How do I reset my password?': go to the login page
Feedback: not helpful
Updated Response to 'How do I reset my password?': go to the login page and click "Forgot Password"
```

## Files
- `support_bot_agent.py` → Main bot implementation
- `faq.txt` → Sample FAQ document
- `support_bot_log.txt` → Log file generated during runs
- `requirements.txt` → Dependency list
- `README.md` → Project documentation

## Design Notes
- Hybrid retrieval: Combines semantic similarity with keyword overlap for resilience on short documents.
- Confidence guardrails: Falls back if QA confidence is low, ensuring clarity over hallucination.
- Feedback handling:  
  - Too vague: Adds context excerpt.  
  - Not helpful: Refines query for specificity.  
- Extensibility: Easily swapped models, chunking strategies, or integrated into FastAPI for deployment.

## Known Limitations
- Extractive QA may return clipped spans; generative models (e.g., BART) could improve fluency.
- Large PDFs may require chunking with overlap.
- Feedback loop is simulated; real-world deployment would integrate user feedback.

## Executive Summary
This project demonstrates how agentic AI workflows can automate customer support by training on domain-specific documents, qualifying queries, and iteratively improving responses.  

It directly aligns with Serri AI’s mission to empower SMEs with affordable, autonomous AI agents that streamline customer engagement across WhatsApp and other channels. The bot showcases:
- Retrieval-Augmented Generation (RAG) for contextual accuracy
- Autonomous feedback-driven improvement
- Transparent logging for trust and reproducibility

By extending this prototype into Serri’s ecosystem (WhatsApp integration, payment flows, lead qualification), it can evolve into a production-ready growth engine for B2C businesses.

## License
MIT

---
