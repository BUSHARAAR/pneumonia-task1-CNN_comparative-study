
# Task 2 – Medical Report Generation using Visual Language Models

## Objective
This task focuses on generating natural-language medical reports from chest X-ray images using a state-of-the-art visual language model (VLM). Unlike Task-1, which addresses image-level classification, Task-2 explores multimodal reasoning by translating visual findings into clinically styled textual descriptions.

## Model Selection
- **Visual Language Model:** MedGemma 1.5 (`google/medgemma-1.5-4b-it`)
- **Reference Classifier:** Vision Transformer (ViT-Tiny), selected as the best-performing model from Task-1 based on AUC and F1-score.

MedGemma is a medical-domain VLM developed under Google Health AI Developer Foundations and is specifically designed for medical imaging tasks, making it suitable for chest X-ray report generation.

## Pipeline Overview
1. Input chest X-ray image from the PneumoniaMNIST test split.
2. ViT-Tiny (Task-1) provides a pneumonia probability and prediction for reference.
3. MedGemma processes the image and a structured prompt to generate a textual medical impression.
4. Outputs include generated reports, corresponding images, and qualitative comparisons.

## Prompting Strategies
Multiple prompting strategies were tested to study their impact on report quality:
- **Concise impression:** Short radiology-style summary.
- **Structured report:** Findings followed by impression.
- **Cautious diagnostic phrasing:** Avoids over-confident conclusions when uncertainty exists.

The effects of these prompts are documented and analyzed in the accompanying report.

## Outputs
- `task2_medgemma.py` – Report generation script.
- `task2 report generation.md` – Detailed qualitative analysis and discussion.
- `outputs/`
  - `task2_samples.json` – Generated reports with metadata.
  - `images/` – Sample chest X-ray images used for evaluation.

## Evaluation
Evaluation is qualitative in nature, as no ground-truth textual reports are available. Generated outputs are compared against:
- Ground truth labels (normal vs pneumonia).
- CNN predictions from ViT-Tiny.
- Visual plausibility and clinical consistency.

## Notes and Limitations
- PneumoniaMNIST images are low resolution (28×28), limiting fine-grained radiological detail.
- Generated reports are not intended for real clinical use.
- VLM outputs may contain generic or uncertain phrasing due to dataset constraints.

## Reproducibility
Execution requires Hugging Face access to MedGemma (public gated repository). Once access is granted, the pipeline can be run end-to-end using the provided script.
