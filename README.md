# Smart MCQ Solver Challenge

## Overview

This project presents a complete machine learning pipeline for solving multiple-choice question (MCQ) answering tasks. Three different approaches are implemented and compared:

1. **TF-IDF + Logistic Regression**
2. **BiGRU + Attention Scratch Model (PyTorch)**
3. **ELECTRA-base Multiple Choice Transformer**

A retrieval-augmented preprocessing step using **TF-IDF Retrieval** and **Cosine Similarity** is used to provide additional context for each question before training.

The best-performing model is automatically selected based on **Validation MAP@3**, retrained on the complete training dataset, and used to generate the final competition submission.

---

# Dataset

The project uses the **Smart MCQ Solver Challenge** dataset.

### Files

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Each question contains:

- Question prompt
- Five answer choices (A–E)
- Correct answer label (training only)

---

# Project Pipeline

## 1. Data Loading

- Load training and test datasets
- Check missing values
- Analyze answer distribution
- Explore prompt and option lengths

---

## 2. Exploratory Data Analysis (EDA)

- Analyze dataset statistics
- Visualize answer distribution
- Analyze text length
- Check duplicate questions
- Perform retrieval quality analysis

---

## 3. Text Preprocessing

- Convert text to lowercase
- Remove extra whitespace
- Normalize text formatting

---

## 4. Retrieval Augmentation

A TF-IDF retriever finds the most similar training question using **Cosine Similarity**.

Retrieved information includes:

- Similar question
- Its correct answer

The retrieved context is appended to every question before model training.

---

## 5. Long Format Conversion

Each MCQ is converted into five training samples.

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

- TF-IDF (1–2 grams)
- 30,000 maximum features
- Logistic Regression
- Balanced class weights

### Validation Results

| Metric | Score |
|---------|-------|
| Validation Loss | **0.4617** |
| Accuracy | **0.9365** |
| F1 Score | **0.8581** |
| MAP@3 | **0.9692** |

---

## Model 2 — BiGRU + Attention Scratch Model

Implemented completely from scratch using **PyTorch**.

### Architecture

- Embedding Layer
- Bidirectional GRU
- Attention Layer
- Dropout
- Fully Connected Layer
- ReLU Activation
- Output Layer

The attention mechanism enables the model to focus on the most informative words in each question-option pair before making the final prediction.

### Validation Results

| Metric | Score |
|---------|-------|
| Validation Loss | **0.0529** |
| Accuracy | **0.9745** |
| F1 Score | **0.9330** |
| MAP@3 | **0.9963** |

This model achieved the highest validation MAP@3 and was selected as the final model for inference.

---

## Model 3 — ELECTRA-base Multiple Choice Transformer

Pretrained transformer using:

- `google/electra-base-discriminator`
- Hugging Face Transformers
- AutoTokenizer
- AutoModelForMultipleChoice

### Validation Results

| Metric | Score |
|---------|-------|
| Validation Loss | **Replace with your score** |
| Accuracy | **Replace with your score** |
| F1 Score | **Replace with your score** |
| MAP@3 | **Replace with your score** |

---

# Model Comparison

| Model | Validation Loss | Accuracy | F1 Score | MAP@3 |
|------|----------------:|---------:|---------:|------:|
| **BiGRU + Attention Scratch Model** | **0.0529** | **0.9745** | **0.9330** | **0.9963** |
| TF-IDF + Logistic Regression | 0.4617 | 0.9365 | 0.8581 | 0.9692 |
| ELECTRA-base Multiple Choice Transformer | **Replace** | **Replace** | **Replace** | **Replace** |

The model with the highest **Validation MAP@3** is automatically selected for generating the final competition predictions.

---

# Training

All three models are trained independently.

Evaluation metrics include:

- Validation Loss
- Validation Accuracy
- Validation F1 Score
- Validation MAP@3

Additional features include:

- Weights & Biases (W&B) experiment tracking
- Automatic best model checkpoint selection
- Error Analysis
- Model Saving

---

# Final Prediction Pipeline

1. Retrain the best-performing model using the complete training dataset.
2. Generate confidence scores for all five answer choices.
3. Rank answer options by confidence.
4. Select the top three predictions.
5. Save predictions as:

```text
submission.csv
```

Example:

| ID | Prediction |
|----|------------|
| 1 | A E C |
| 2 | B E A |

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Seaborn
- Graphviz
- Joblib
- Weights & Biases (W&B)

---

# Installation

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm torch torchvision transformers datasets accelerate sentencepiece wandb joblib graphviz
```

---

# Running the Notebook

1. Download the Smart MCQ Solver Challenge dataset.
2. Place the dataset inside the Kaggle input directory.
3. Run all notebook cells.
4. The notebook will:

- preprocess the data
- perform exploratory data analysis
- retrieve contextual information
- train all three models
- compare model performance
- automatically select the best model
- perform error analysis
- save trained models
- generate `submission.csv`

---

# Future Improvements

- Cross-validation
- BM25 retrieval
- Dense Retrieval
- Sentence Transformer retrieval
- Larger Transformer Models (RoBERTa-large, DeBERTa-v3)
- Model ensembling
- Knowledge distillation
- Hard negative mining
- Hyperparameter optimization

---

# Author

Developed as a solution for the **Smart MCQ Solver Challenge**, demonstrating classical machine learning, deep learning from scratch using **BiGRU with Attention**, retrieval-augmented learning, and transformer-based multiple-choice question answering using **ELECTRA-base**.