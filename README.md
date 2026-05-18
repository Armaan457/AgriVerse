# Multimodal Crop Assistant

A **compact, research grade AI system** for diagnosing plant leaf diseases and providing **knowledge grounded agricultural guidance**. This system blends **computer vision**, **large language models**, and **knowledge graphs** to deliver accurate, explainable, and context aware plant disease assistance.

---

## Overview

The Assistant allows users to upload a leaf image or simply ask a question about a crop.
The system then:

1. **Identifies the crop and potential disease** (from image or text).
2. **Retrieves scientific facts** from a structured knowledge graph.
3. **Generates a grounded, evidence-based answer** using an LLM.
4. **(Optional)** Produces explainability heatmaps showing areas influencing the model’s decision.

## Core Pipeline

### Vision/Text Diagnosis Component

#### **Image Input**

A trained vision model classifies the leaf image into:

  * **Crop type**
  * **Disease**

* Optionally generates **Grad-CAM heatmaps**

#### **Text Only Input**

If no image is provided:

* A **fuzzy matching algorithm** extracts crop and disease mentions from the user’s query.

---

### Knowledge Grounded QA Component

Once the disease/crop are known:

1. **Neo4j Knowledge Graph Query**

   Retrieves nodes such as:

     * Disease symptoms
     * Causal pathogens
     * Affected crops
     * Available treatments

   Along with their relationships
   
2. **LLM Answer Synthesis**

    The language model receives:
    * User question
    * Predicted crop/disease
    * Structured facts from the graph
   
   This Produces a **precise, scientifically grounded answer**.

---

## Evaluation Results

### Vision Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | 99% |
| Precision | 99% |
| Recall | 99% |
| F1 Score | 99% |

### Final Output Correctness

| Evaluation Method | Score |
|------------------|--------|
| LLM-as-a-Judge | 9.1/10 |
