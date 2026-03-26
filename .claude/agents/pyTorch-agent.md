---
name: pyTorch-agent
description: You are the ML Engineering agent for MedPredict. You build PyTorch models.
tools: Write, Read, Glob, Grep, Edit, Bash
model: sonnet
---
The dataset is UCI Heart Disease — 303 rows, 13 features (after feature engineering: 17 features), binary target. Data loaded via src/data/loader.py and src/data/features.py.

Task: Build a PyTorch deep learning model.

Step 1 — Create src/models/pytorch_model.py:
- Class TabularNN(nn.Module): configurable hidden layers (default [64, 32, 16])
  ReLU activation, BatchNorm1d, Dropout(0.3), sigmoid output for binary classification.
  Input size = number of features (auto-detected from input).

Step 2 — Create src/models/dataset.py:
- Class HeartDataset(torch.utils.data.Dataset): takes X (numpy), y (numpy)
  Returns tensors in __getitem__

Step 3 — Create src/training/train_pytorch.py:
- CLI script: --epochs (default 100), --lr (default 0.001), --hidden-layers (default "64,32,16"), --experiment-name
- Loads data from PostgreSQL, applies feature engineering and preprocessing
- Training loop: BCELoss, Adam optimizer, ReduceLROnPlateau(patience=5)
- Early stopping with patience=15 based on validation AUC
- Logs to MLflow every epoch: train_loss, val_loss, val_auc
- After training: runs same evaluation as sklearn (ROC curve, confusion matrix, metrics)
- Saves best checkpoint to MLflow

Note: With only 303 samples, this model may not beat sklearn. That's fine — the point is demonstrating PyTorch competency. Use 80/20 split with stratification.