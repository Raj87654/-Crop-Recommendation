# 🌱 Crop Recommendation System using XGBoost

## 📌 Project Overview

This project uses Machine Learning to recommend the most suitable crop for cultivation based on soil nutrients and environmental conditions. The model analyzes various agricultural parameters and predicts the best crop that can be grown under the given conditions.

The objective of this project is to assist farmers and agricultural stakeholders in making informed crop selection decisions, improving productivity and resource utilization.

---

## 🚀 Features

* Data preprocessing and exploration
* Crop recommendation using Machine Learning
* XGBoost Classification Model
* Model evaluation using Accuracy Score
* Classification Report
* Confusion Matrix Visualization
* Interactive crop prediction based on user inputs

---

## 📊 Dataset Information

The dataset contains agricultural and environmental parameters:

| Feature     | Description                 |
| ----------- | --------------------------- |
| N           | Nitrogen content in soil    |
| P           | Phosphorous content in soil |
| K           | Potassium content in soil   |
| Temperature | Temperature in °C           |
| Humidity    | Relative humidity (%)       |
| pH          | Soil pH value               |
| Rainfall    | Rainfall in mm              |
| Label       | Recommended Crop            |

### Supported Crops

* Rice
* Maize
* Chickpea
* Kidney Beans
* Pigeon Peas
* Moth Beans
* Mung Bean
* Black Gram
* Lentil
* Pomegranate
* Banana
* Mango
* Grapes
* Watermelon
* Muskmelon
* Apple
* Orange
* Papaya
* Coconut
* Cotton
* Jute
* Coffee

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn

---

## 📈 Model Training

The dataset is divided into training and testing sets using an 80:20 ratio.

Model Used:

* XGBoost Classifier

Evaluation Metrics:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📋 Project Workflow

1. Load Dataset
2. Data Exploration
3. Data Preprocessing
4. Label Encoding
5. Train-Test Split
6. XGBoost Model Training
7. Model Evaluation
8. Crop Prediction
9. Visualization of Results

---

## 🎯 Example Prediction

Input:

Nitrogen (N): 90
Phosphorous (P): 42
Potassium (K): 43
Temperature: 20.87
Humidity: 82.00
pH: 6.50
Rainfall: 202.93

Predicted Crop:

Rice

---

## 📂 Project Structure

Crop-Recommendation/

├── data/
│   └── Crop_recommendation.csv

├── notebooks/
│   └── Crop_Recommendation.ipynb

├── models/
│   └── crop_model.pkl

├── src/
│   ├── train_model.py
│   └── predict.py

├── images/
│   └── confusion_matrix.png

├── requirements.txt

├── README.md

└── .gitignore

---

## ▶️ Installation

Clone the repository:

git clone https://github.com/your-username/Crop-Prediction-model.git

Navigate to the project folder:

cd Crop-Prediction-model

Install dependencies:

pip install -r requirements.txt

Run the project:

python predict.py

---

## 📌 Future Improvements

* Streamlit Web Application
* Real-Time Weather API Integration
* Fertilizer Recommendation System
* Crop Yield Prediction
* Disease Detection Module
* Mobile Application Integration

---

## 👨‍💻 Author

Rajkumar M
