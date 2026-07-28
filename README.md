#  Smart MCQ Solver Challenge

An end-to-end Deep Learning project for the **Smart MCQ Solver Kaggle Competition**. This project predicts the **Top-3 most probable answers** for multiple-choice questions using Machine Learning, Deep Learning, and Transformer-based approaches, evaluated with the official **MAP@3** metric.

---

##  Kaggle Performance

- **Competition:** Smart MCQ Solver Challenge
- **Evaluation Metric:** MAP@3
- **Public Kaggle Score:** **0.753**

---

##  Dataset

| Dataset | Samples |
|----------|---------|
| Training | 2,000 |
| Testing | 500 |

Each question contains:
- Question Prompt
- Five Answer Options (A–E)
- Correct Answer (Training only)

---

##  Exploratory Data Analysis

The notebook performs:
- Dataset overview
- Missing value analysis
- Question length analysis
- Answer distribution analysis
- Option length comparison
- Duplicate prompt detection and removal
- Class balance analysis

---

##  Data Preprocessing

- Text cleaning (lowercase, special character removal, URL removal)
- Duplicate prompt removal
- Long-format dataset creation
- TF-IDF feature extraction
- ELECTRA tokenization
- Train-validation split
- Class weight computation

---

##  Models Implemented

### 1. TF-IDF + Logistic Regression (Baseline)
Traditional machine learning baseline using TF-IDF features and Logistic Regression.

### 2. BiGRU + Attention (Scratch)
A custom PyTorch model built from scratch using:
- Embedding Layer
- Bidirectional GRU
- Attention Layer
- Dense Network

### 3. ELECTRA-base (Pretrained)
Fine-tuned **Google ELECTRA-base** using Hugging Face Transformers for multiple-choice question answering.

### 4. TF-IDF + PyTorch MLP
A custom neural network trained on TF-IDF features.

---

##  Model Performance

| Model | Accuracy | F1 Macro | MAP@3 |
|--------|----------:|----------:|-------:|
| TF-IDF + Logistic Regression | 0.8369 | 0.6096 | 0.9692 |
| TF-IDF + PyTorch MLP | 0.9744 | 0.9594 | 0.9735 |
| BiGRU + Attention | 0.9699 | 0.9259 | 0.9915 |
| ELECTRA-base | **0.9943** | **0.9943** | **0.9972** |

 **Best Validation Model:** ELECTRA-base

---

##  Error Analysis

The notebook includes:
- Classification Report
- Confusion Matrix
- Validation Performance Analysis
- Model Comparison

---

##  Final Output

The best-performing ELECTRA model is used to generate the final Kaggle submission file:

```text
submission_electra_mc.csv
```

---

##  Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Graphviz
- Weights & Biases (W&B)

---
##  Future Improvements

- Model Ensembling
- K-Fold Cross Validation
- Hyperparameter Optimization
- Test-Time Augmentation (TTA)
- Larger Transformer Models

---

##  Author

**Hariom Patel**

IIT Madras BS Degree Program

---

##  Acknowledgements

- Kaggle
- IIT Madras
- Hugging Face
- PyTorch
- Weights & Biases

---

 If you found this project helpful, consider giving it a **Star** on GitHub.