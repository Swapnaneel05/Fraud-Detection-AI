SMS Fraud Detection System (RAG-Based)
🎓 Intel Unnati Program - GenAI Project
This repository contains a Retrieval-Augmented Generation (RAG) system designed to detect and classify fraudulent SMS messages. By leveraging the Zephyr 7B Beta model and a custom vector database of known fraud patterns, this system provides highly accurate, context-aware fraud analysis.

📌 Project Overview
Traditional fraud detection often relies on static rules or simple keywords. This project elevates detection by using Generative AI.

By using RAG, the system doesn't just rely on the LLM's pre-trained knowledge; it "retrieves" similar fraudulent patterns from a provided CSV dataset to provide a grounded, reasoned classification of whether an incoming message is a threat.

🛠️ Tech Stack
Model: Zephyr-7b-beta (via Hugging Face)

Orchestration: Python, LangChain (or equivalent RAG framework)

Vector Store: FAISS / ChromaDB (for storing SMS embeddings)

Data: CSV dataset of labeled Fraud/Ham SMS messages

Optimization: Optimized for Intel-based environments as part of the Intel Unnati curriculum.

🚀 Features
Semantic Search: Uses embeddings to find similar fraud cases even if keywords don't match exactly.

Reasoned Classification: Instead of a simple "Fraud/Not Fraud" label, the model explains why a message is suspicious.

Reduced Hallucinations: RAG ensures the model stays grounded in the provided dataset.
