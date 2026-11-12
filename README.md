# 💎 Diamond Price Prediction

## 📌 Project Overview

This project predicts the price of diamonds using Machine Learning regression algorithms. It analyzes diamond characteristics such as carat, cut, color, clarity, and dimensions to estimate diamond prices.

The project also includes a Streamlit web application that allows users to enter diamond details and receive price predictions.

## 🎯 Project Objectives

- Analyze diamond features and their relationship with price.
- Perform data cleaning and preprocessing.
- Build and compare multiple regression models.
- Select a suitable machine learning model.
- Deploy the prediction model using Streamlit.

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
- Git and GitHub

## 🤖 Machine Learning Algorithms

The following regression algorithms were compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors (KNN) Regressor

The trained KNN pipeline is saved and used by the Streamlit application.

## 📊 Model Evaluation

The models were compared using regression evaluation metrics.

Refer to `model_comparison.csv` for the model comparison results.

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

## 🌐 Application

The application allows users to enter diamond characteristics and obtain an estimated diamond price.

## 👩‍💻 Author

**Alekya**

Machine Learning | Data Science | Python