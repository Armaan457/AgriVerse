# 🌿 AgriVerse

An end-to-end AI-powered crop disease diagnosis and agricultural assistant that combines **computer vision**, **knowledge graphs**, and **large language models** to deliver accurate, explainable, and scientifically grounded recommendations for farmers.

**Live Demo:** [Try AgriVerse](https://huggingface.co/spaces/Armaan457/AgriVerse)

---

## Key Features

- Diagnose plant diseases from images
- Answer agricultural queries using natural language
- Knowledge graph grounded responses 
- Explainable predictions with Grad-CAM visualizations
- Supports both image-based and text-only interactions

---


## Project Highlights

- Achieved **99% classification accuracy** on plant disease detection.
- Combines **vision models, knowledge graphs, and LLMs** into a single multimodal pipeline.
- Generates **grounded responses** using structured agricultural knowledge instead of relying solely on LLM reasoning.
- Provides **visual explanations** through Grad-CAM for improved model interpretability.

---

## Model Performance

### Disease Classification

| Metric | Score |
|---------|------:|
| Accuracy | 99% |
| Precision | 99% |
| Recall | 99% |
| F1 Score | 99% |

### End-to-End Response Quality

| Evaluation | Score |
|------------|------:|
| LLM-as-a-Judge | 9.1 / 10 |

Scored on four dimensions: **Faithfulness** (consistency with the knowledge graph), **Scientific Correctness**, **Completeness**, and **Hallucination Control**.

---

## Dataset

Trained on a large-scale crop disease dataset comprising **130000+ images, 16 crop species and 33 disease categories**.

## Tech Stack

Built with **PyTorch** and **Hugging Face Transformers** using a fine-tuned **Vision Transformer (ViT)**. Responses are grounded through a **Neo4j Knowledge Graph** queried using the **Cypher Query Language**, with **Grad-CAM** for visual explainability.

