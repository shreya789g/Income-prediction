# Income-prediction
Employee Salary Prediction Using Machine Learning

A machine learning–based web application that predicts the annual salary of software developers using insights from the Stack Overflow Developer Survey 2019 dataset.
This project uses Python, Scikit-learn, Pandas, NumPy, Matplotlib, and a Streamlit UI for deployment.

📌 Project Overview

This project focuses on building a predictive model that estimates developer salaries based on key factors such as Country, Education Level, and Years of Professional Experience.
The goal is to demonstrate how machine learning can be applied to real-world survey data to extract insights and build an interactive salary prediction tool.

The workflow includes data cleaning, preprocessing, model development, evaluation, and the creation of a Streamlit-based web app for real-time predictions.

📂 Dataset

The dataset is taken from the Stack Overflow Developer Survey 2019, a globally recognized dataset used widely for industry and academic research.

Dataset includes:

Country

Education Level

Years of Coding Experience

Annual Salary

Developer demographics

🧠 Machine Learning Models Used

The following algorithms were implemented and compared:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

The best-performing model was selected using GridSearchCV for hyperparameter optimization.

🛠 Tech Stack

Languages & Libraries:

Python (3.x)

Pandas

NumPy

Scikit-Learn

Matplotlib

Joblib

Deployment Framework:

Streamlit

Tools Used:

Jupyter Notebook

Visual Studio Code

Git / GitHub

⚙️ Project Workflow

Data Cleaning & Preprocessing

Removing missing or invalid entries

Label encoding categorical variables (Country, EdLevel)

Converting text-based experience values into numeric

Handling outliers using boxplots

Selecting relevant features

Model Training & Testing

Training multiple ML models

Comparing model metrics (MSE, MAE)

Hyperparameter tuning with GridSearchCV

Model Deployment

Saving the trained model using Joblib

Creating a Streamlit interface

Building interactive input fields

Displaying predicted salary in real time

📊 Visualizations

The project uses Matplotlib to generate:

Salary distribution graphs

Boxplots for outlier detection

Feature analysis plots

🚀 How to Run the Project Locally
1. Clone the Repository
git clone <your-repo-link>
cd <your-project-folder>

2. Install Dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run app.py

📌 Project Structure
├── data/
│   └── survey_results_public.csv
├── notebooks/
│   └── EDA_and_Model.ipynb
├── app.py
├── model.pkl
├── requirements.txt
└── README.md

✨ Features

Predicts developer salary based on inputs

User-friendly web interface

Supports multiple countries

Fast and accurate predictions

End-to-end machine learning pipeline

📉 Limitations

Predictions depend on the 2019 dataset, may not reflect current trends

Only suitable for software developer roles

Dataset biases may affect accuracy

Does not include deep learning models


📚 References

[1] Stack Overflow Developer Survey, 2019
[2] Scikit-learn Documentation
[3] NumPy Documentation
[4] Pandas Documentation
[5] Streamlit Official Documentation
[6] Jupyter Notebook Documentation
[7] Python Language Reference
