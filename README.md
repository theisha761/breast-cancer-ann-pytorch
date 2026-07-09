# Breast Cancer Classification using Artificial Neural Network (PyTorch)

## Project Overview

This project implements a complete binary classification pipeline using an Artificial Neural Network (ANN) built from scratch in **PyTorch**.

The model predicts whether a breast tumor is **malignant** or **benign** using the Breast Cancer Wisconsin Diagnostic Dataset from scikit-learn.

The primary objective of this project was to understand the complete deep learning workflow rather than relying on high-level libraries.

---

## Dataset

Dataset:
- Breast Cancer Wisconsin Diagnostic Dataset
- Source: scikit-learn
- Samples: 569
- Features: 30 numerical features
- Classes:
  - Benign
  - Malignant

---

## Technologies Used

- Python
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

---

## Project Pipeline

1. Load dataset
2. Train-test split
3. Feature scaling using StandardScaler
4. Convert NumPy arrays to PyTorch tensors
5. Create custom Dataset class
6. Create DataLoaders for mini-batch training
7. Build ANN using nn.Sequential
8. Train using mini-batch gradient descent
9. Evaluate model
10. Save trained model

---

## Neural Network Architecture

Input (30 Features)

↓

Linear (30 → 16)

↓

Batch Normalization

↓

ReLU

↓

Linear (16 → 8)

↓

Batch Normalization

↓

ReLU

↓

Linear (8 → 1)

↓

Sigmoid

---

## Techniques Implemented

- Mini-batch Gradient Descent
- GPU Training (CUDA)
- Feature Scaling
- Batch Normalization
- L2 Regularization (Weight Decay)
- Binary Cross Entropy Loss
- SGD Optimizer
- Model Saving
- Evaluation Mode (`model.eval()`)

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix

---

## Final Results

| Metric | Score |
|---------|-------|
| Test Accuracy | 95.6% |
| Precision | 92.4% |
| Recall | 100% |
| F1 Score | 96.0% |
| ROC-AUC | 0.993 |

The model achieved **100% recall**, meaning it successfully identified every positive case in the test set while maintaining high overall precision.

---

## Future Improvements

- Early Stopping
- Learning Rate Scheduler
- Hyperparameter Optimization
- Cross Validation
- TensorBoard Logging
- Model Deployment using Streamlit or FastAPI

## Saved Model

The trained model weights are included as:

`breast_cancer_ann.pth`

To load the model:

```python
model = MyNN()
model.load_state_dict(torch.load("breast_cancer_ann.pth"))
model.eval()

---
