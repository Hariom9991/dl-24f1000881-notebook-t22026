# Smart MCQ Solver Challenge

## Overview

This project presents a complete machine learning pipeline for solving multiple-choice question (MCQ) answering tasks. Three different approaches are implemented and compared:

1. **TF-IDF + Logistic Regression**
2. **Scratch Neural Ranker (PyTorch)**
3. **DistilBERT Multiple Choice Transformer**

A retrieval-augmented preprocessing step is used to provide additional context for each question before training.

The best-performing model is automatically selected based on **Validation MAP@3**, retrained on the complete training dataset, and used to generate the final competition submission.

---

## Dataset

The project uses the **Smart MCQ Solver Challenge** dataset.

### Files

* `train.csv`
* `test.csv`
* `sample_submission.csv`

Each question contains:

* Question prompt
* Five answer choices (A–E)
* Correct answer label (training only)

---

# Project Pipeline

## 1. Data Loading

* Load training and test datasets
* Check missing values
* Explore answer distribution
* Analyze prompt and option lengths

---

## 2. Text Preprocessing

* Convert text to lowercase
* Remove extra spaces
* Normalize formatting

---

## 3. Retrieval Augmentation

A TF-IDF retriever finds the most similar training question.

Retrieved information includes:

* Similar question
* Its correct answer

This retrieved context is appended to each question before training.

---

## 4. Long-Format Conversion

Each MCQ is converted into five separate training samples.

Example:

Question + Retrieved Context + Option A

Question + Retrieved Context + Option B

...

Question + Retrieved Context + Option E

Positive label = Correct option

Negative label = Incorrect option

---

# Models

## Model 1 — TF-IDF + Logistic Regression

### Features

* TF-IDF (1–2 grams)
* 30,000 maximum features
* Logistic Regression
* Balanced class weights

### Validation Results

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **0.9365** |
| F1       | **0.8581** |
| Loss     | **0.4617** |
| MAP@3    | **0.9692** |

This achieved the **highest MAP@3** among all models.

---

## Model 2 — Scratch Neural Ranker

Implemented completely from scratch using PyTorch.

Architecture:

* Embedding layer
* Average pooling
* Fully connected layer
* ReLU
* Dropout
* Output layer

### Validation Results

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **0.8975** |
| F1       | **0.6623** |
| Loss     | **0.2241** |
| MAP@3    | **0.9471** |

---

## Model 3 — DistilBERT Multiple Choice

Pretrained transformer using:

* `distilbert-base-uncased`
* Hugging Face Transformers
* AutoModelForMultipleChoice

### Validation Results

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **0.8950** |
| F1       | **0.8896** |
| Loss     | **0.4363** |
| MAP@3    | **0.9283** |

---

# Model Comparison

| Model                        |   Accuracy |         F1 |      MAP@3 |
| ---------------------------- | ---------: | ---------: | ---------: |
| TF-IDF + Logistic Regression | **0.9365** |     0.8581 | **0.9692** |
| Scratch Neural Ranker        |     0.8975 |     0.6623 |     0.9471 |
| DistilBERT Multiple Choice   |     0.8950 | **0.8896** |     0.9283 |

The TF-IDF + Logistic Regression model achieved the highest validation MAP@3 and was selected for final inference.

---

# Training

The notebook trains all three models independently.

Evaluation metrics include:

* Validation Loss
* Accuracy
* F1 Score
* MAP@3

Weights & Biases (W&B) is used for experiment tracking.

---

# Final Prediction Pipeline

1. Retrain the best model on the full training dataset.
2. Generate scores for all five options.
3. Rank options by confidence.
4. Keep the top three predictions.
5. Save results as:

```
submission.csv
```

Example:

| ID | Prediction |
| -- | ---------- |
| 1  | A E C      |
| 2  | B E A      |

---

# Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Matplotlib
* Seaborn
* Weights & Biases

---

# Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers sentencepiece wandb
```

---

# Running the Notebook

1. Download the competition dataset.
2. Place it inside the Kaggle input directory.
3. Run all notebook cells.
4. The notebook will:

   * preprocess the data,
   * train all models,
   * compare performance,
   * select the best model,
   * generate `submission.csv`.


# Future Improvements

* Cross-validation
* Better retrieval using Sentence Transformers
* BM25 retrieval
* Larger pretrained models (RoBERTa, DeBERTa)
* Model ensembling
* Knowledge distillation
* Hard negative mining

---

# Author

Developed as a solution for the **Smart MCQ Solver Challenge**, demonstrating classical machine learning, neural networks, transformer-based models, and retrieval-augmented ranking for multiple-choice question answering.
