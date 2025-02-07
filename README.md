Project Overview:

This Python-based fraud detection system leverages machine learning to identify potentially fraudulent transactions. It extracts data from an SQL database, performs preprocessing, and applies predictive analytics using Random Forest and Gradient Boosting classifiers. The system is integrated with Streamlit for real-time fraud predictions and automated alerts.

Key Features:

✅ Loads transactional data from an SQLite database

✅ Performs feature engineering (e.g., transaction hour, high-amount flag, customer spending behavior)

✅ Trains multiple classification models (RandomForest & GradientBoosting) for fraud detection✅ Evaluates models with accuracy scores and confusion matrices

✅ Deploys an interactive UI using Streamlit for real-time fraud predictions✅ Sends fraud alerts via API when suspicious transactions are detected

Tech Stack:

Python (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)

Machine Learning (Random Forest, Gradient Boosting)

Database (SQLite for storing transaction records)

Data Visualization (Matplotlib, Seaborn for insights)

Streamlit (Web-based fraud prediction UI)

API Integration (Sends fraud alerts via HTTP requests)


Implementation Steps:

Step 1: Load Data from SQL

Extracts transaction records from an SQLite database.

Step 2: Data Preprocessing & Feature Engineering

Converts transaction time into hour-based features.

Flags high-amount transactions based on the 95th percentile.

Computes average transaction amount per customer.

Calculates total number of transactions per customer.

Step 3: Train and Evaluate Machine Learning Models

Trains Random Forest and Gradient Boosting classifiers.

Splits data into training/testing sets (80/20 ratio).

Evaluates model performance with accuracy scores & confusion matrices.

Step 4: Fraud Prediction and Real-Time Alerting

Uses Streamlit UI to accept transaction details for fraud prediction.

Calls fraud detection model to classify transactions as fraudulent or legitimate.

If fraud is detected, triggers an API alert to notify stakeholders.

How to Run the Project

1️⃣ Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn sqlite3 streamlit requests

2️⃣ Run the Streamlit app:

streamlit run fraud_detection.py

3️⃣ Enter transaction details and predict fraud!
