# Crop Recommendation Model

This repository contains a machine learning pipeline for crop recommendation using XGBoost.

## Directory Structure
```text
Crop-Recommendation/
│
├── data/
│   └── Crop_recommendation.csv
│
├── notebooks/
│   └── Crop_Recommendation.ipynb
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── crop_model.pkl
│
├── images/
│   ├── confusion_matrix.png
│   └── crop_distribution.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   cd src
   python train_model.py
   ```
   This will train the XGBoost classifier, save the model to the `models/` directory, and generate evaluation plots in the `images/` directory.

3. Run predictions:
   ```bash
   cd src
   python predict.py
   ```
   This script provides an interactive CLI tool for predicting crops based on soil and weather parameters.