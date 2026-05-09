# Pixels-to-Predictions: Multi-Modal Science Question Answering with SmolVLM

**Team:** Ananya and Krishan

**Competition:** Pixels-to-Predictions (Kaggle Deep Learning Final Competition)

**Base Model:** `HuggingFaceTB/SmolVLM-500M-Instruct`

**Final Best Leaderboard Score:** **0.88933** (`sub_score_average.csv`)

---

# Overview

This repository contains our complete experimental pipeline, trained adapters, notebooks, submission files, and report for the **Pixels-to-Predictions** multi-modal science question answering competition.

The task involves answering science multiple-choice questions using:

* images
* textual context
* hints
* metadata
* scientific reasoning

We explored a wide range of parameter-efficient fine-tuning and inference strategies using SmolVLM.

---

# Main Techniques Explored

## Training Strategies

* LoRA fine-tuning
* DoRA fine-tuning
* Attention-only adaptation
* Attention + MLP adaptation
* Native-resolution image training
* High-resolution image training
* Hard-sample mining
* Metadata-aware prompting
* Full-answer supervision
* Train+validation combined training

## Inference Strategies

* Letter log-probability scoring
* Candidate-token constrained decoding
* Prompt engineering
* Confidence-based routing
* Ensemble averaging
* Test-time augmentation (TTA)
* Metadata prompting

---

# Hardware and Environment

All experiments were conducted using:

* Google Colab Student Subscription
* NVIDIA A100 40GB GPUs

### Software

* Python 3.12
* PyTorch 2.x
* Transformers 4.57.1
* PEFT 0.18.0
* bitsandbytes 0.45.x
* Accelerate 1.7.x

Approximate total compute:

* ~120 GPU-hours

---

# Repository Structure

```text
.
├── notebooks/
├── checkpoints/
├── submissions/
├── report/
├── requirements.txt
└── README.md
```

## notebooks/

Contains all major experimental notebooks and ablation studies.

## checkpoints/

Contains final trained LoRA/DoRA adapters.

## submissions/

Contains all generated Kaggle submission CSV files.

## report/

Contains the final report and LaTeX source.

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Krishan101/kaggle-dl-pixels-to-predictions.git
cd kaggle-dl-pixels-to-predictions
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

The competition dataset is downloaded using `kagglehub`.

## Kaggle Authentication

```python
import kagglehub
kagglehub.login()
```

## Download Dataset

```python
COMP_DIR = kagglehub.competition_download("pixels-to-predictions")
```

Expected structure:

```text
pixels-to-predictions/
├── train.csv
├── val.csv
├── test.csv
├── sample_submission.csv
├── images/
```

---

# Important Notes About Checkpoints

Large model checkpoints are not fully stored directly in GitHub due to repository size constraints.

The repository includes:

* final lightweight adapters
* training notebooks
* inference notebooks
* submission CSVs
* complete report and reproducibility pipeline

Intermediate training checkpoints were intentionally excluded.

---

# Reproducing Main Experiments

## 1. Zero-Shot Baseline

Notebook:

```text
notebooks/ZeroShot_Correct_Baseline.ipynb
```

Expected leaderboard score:

```text
0.69818
```

---

## 2. Native-Resolution LoRA Training

Notebook:

```text
notebooks/LoRA-v2-NativeRes-Train.ipynb
```

Expected leaderboard score:

```text
0.75251
```

---

## 3. High-Resolution Training

Notebook:

```text
notebooks/LoRA v6 HighRes512 Training.ipynb
```

Expected leaderboard score:

```text
0.76257
```

---

## 4. Full-Layer Attention + MLP Adaptation

Notebook:

```text
notebooks/LoRA v3 Full-Layer Adaptation (Attention MLP).ipynb
```

Expected leaderboard score:

```text
0.75452
```

---

## 5. LoRA v9 Best Single Model

Notebook:

```text
notebooks/LoRA-v9-r8-all7-flatLR-trainval.ipynb
```

Key configuration:

* LoRA rank = 8
* Attention + MLP targets
* Flat learning-rate schedule
* Metadata prompting
* Train+validation combined training

Expected leaderboard score:

```text
0.86116
```

---

## 6. Metadata Prompt Inference

Notebook:

```text
notebooks/LoRA-v9-Inference-MetadataPrompt.ipynb
```

Expected leaderboard score:

```text
0.85311
```

---

# Final Ensemble Reproduction

Our best final submission:

```text
submissions/sub_score_average.csv
```

achieved:

```text
0.88933
```

The final ensemble combined predictions from multiple independently trained adapters using score averaging.

Main ensemble components included:

* metadata-prompted models
* attention+MLP LoRA adapters
* DoRA adapters
* diverse random seeds
* inference-time averaging

## Ensemble-Related Submission Files

```text
submissions/sub_ensemble_all_FG.csv
submissions/sub_ensemble_all_noC.csv
submissions/sub_ensemble_no_C.csv
submissions/sub_score_average.csv
```

---

# Running Inference

General inference workflow:

```python
pred, scores = predict_score_one(row)
```

Inference uses:

* constrained candidate-token scoring
* log-probability ranking
* metadata-aware prompting
* ensemble averaging

Submission generation:

```python
submission.to_csv("submission.csv", index=False)
```

Expected Kaggle submission format:

```text
id,answer
sample_001,2
sample_002,0
```

---

# Major Experimental Findings

## Strong Improvements

* DoRA outperformed standard LoRA
* Attention + MLP adaptation improved reasoning
* Metadata prompting significantly improved inference
* Ensemble averaging produced the largest gains

## Moderate Improvements

* Native-resolution image training
* High-resolution image inputs
* Prompt-template optimization
* Hard-sample fine-tuning

## Negative / Unstable Results

* Chat-template masking experiments
* Aggressive TTA
* Overly complex prompt chains

---

# Parameter-Efficient Fine-Tuning

All experiments were conducted under the competition parameter-budget constraints using parameter-efficient fine-tuning methods.

Target modules explored included:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

---

# Requirements

All required libraries and package versions are listed in:

```text
requirements.txt
```

---

# Authors

* Ananya Nadig (an4968)
* Krishan Kumar Gupta (kg3868)

---

# Repository

[https://github.com/Krishan101/kaggle-dl-pixels-to-predictions](https://github.com/Krishan101/kaggle-dl-pixels-to-predictions)
