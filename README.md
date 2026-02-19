# Pneumonia Detection and Medical Report Generation using Deep Learning and Visual Language Models

## Overview
This repository presents a two-stage study on automated pneumonia analysis from chest X-ray images:

- **Task 1:** Comparative evaluation of deep learning models for pneumonia classification.
- **Task 2:** Medical report generation using a visual language model (VLM).

The work demonstrates both **predictive performance** and **interpretability through multimodal AI**, combining image classification with natural-language medical reasoning.

---

## Task 1 – Pneumonia Detection (CNN / Transformer / Mamba)

### Objective
Develop and evaluate multiple deep learning architectures for binary pneumonia detection from chest X-ray images, emphasizing rigorous experimental methodology and comparative analysis.

### Models Evaluated
- Simple CNN
- ResNet-18
- EfficientNet-B0
- Vision Transformer (ViT-Tiny)
- MambaNet (state-space model)

### Dataset
- **PneumoniaMNIST** (MedMNIST v2)
- Standard train/validation/test splits

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Area Under the ROC Curve (AUC)
- Confusion matrix and ROC analysis

### Key Outcome
ViT-Tiny achieved the best overall discrimination performance (highest AUC and F1-score) and was selected as the reference model for Task-2.

---

## Task 2 – Medical Report Generation (Visual Language Model)

### Objective
Generate natural-language medical impressions from chest X-ray images using a medical-domain visual language model.

### Model
- **MedGemma 1.5** (`google/medgemma-1.5-4b-it`)
- Reference classifier: ViT-Tiny (Task-1 best model)

### Highlights
- Image-to-text medical report generation
- Prompt engineering and qualitative analysis
- Comparison of VLM outputs with ground truth labels and CNN predictions
- Analysis of failure and ambiguous cases

---

## Repository Structure

---

## Reproducibility
- Python, PyTorch, Hugging Face Transformers
- Task-2 requires Hugging Face access to MedGemma (public gated model)
- All experiments are reproducible using the provided scripts

---

## Disclaimer
This project is intended for academic and research purposes only.  
Generated predictions and reports are **not suitable for clinical diagnosis**.

---

## Author
**Bushara A. R.**  
Deep Learning | Medical Image Analysis | Explainable & Multimodal AI
