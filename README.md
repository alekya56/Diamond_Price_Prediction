# 💎 Diamond Price Prediction

## 📌 Project Overview

This project predicts diamond prices using Machine Learning regression algorithms. It analyzes diamond characteristics such as carat, cut, color, clarity, depth, table, and dimensions to estimate diamond prices.

The project includes an interactive Streamlit web application where users can enter diamond characteristics and receive instant price predictions.

## 🎯 Project Objectives

- Explore diamond features and their relationship with price.
- Perform data cleaning and preprocessing.
- Build and evaluate multiple regression models.
- Compare model performance using evaluation metrics.
- Use the KNN Regressor for diamond price prediction.
- Deploy the machine learning application using Streamlit.

## 📂 Project Structure

```text
Diamond_Price_Prediction/
│
├── app.py
├── description.txt
├── requirements.txt
├── model_comparison.csv
├── README.md
│
└── models/
    └── diamond_knn_pipeline.pkl
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Altair
- Git and GitHub

## 📊 Dataset Information

The project uses a diamond dataset containing diamond characteristics and their corresponding prices.

Features include:

- Carat
- Cut
- Color
- Clarity
- Depth
- Table
- Length (x)
- Width (y)
- Height (z)

**Target variable:** Price

## 🤖 Machine Learning Algorithms

The following regression algorithms were implemented and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors (KNN) Regressor

The trained KNN pipeline is saved as a Pickle file and used by the Streamlit application for price predictions.

## 📈 Model Evaluation Results

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Random Forest Regressor | 272.65 | 538.46 | 0.9814 |
| Decision Tree Regressor | 362.78 | 733.00 | 0.9655 |
| KNN Regressor | 408.31 | 765.01 | 0.9624 |
| Linear Regression | 731.74 | 1122.99 | 0.9189 |

**Best Performing Model:** Random Forest Regressor

Random Forest achieved the highest R² score of 0.9814 (98.14%) and the lowest MAE and RMSE among the four models.

**Final Prediction Model:** KNN Regressor, as required by the project assignment.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/alekya56/Diamond_Price_Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Diamond_Price_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## 🌐 Live Application

Try the deployed application here:

[Diamond Price Prediction – Streamlit App](https://diamondpriceprediction-cgrykhuoursikkmgrjudpp.streamlit.app/)

## 📥 Model Comparison Report

The application provides an option to download the model comparison report containing the evaluation results of all four regression algorithms.

## 👩‍💻 Author

**Alekya**

Machine Learning | Data Science | Python