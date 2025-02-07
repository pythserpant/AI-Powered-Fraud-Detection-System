import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import requests
import json

# Step 1: Load Data from SQL
def load_data():
    conn = sqlite3.connect("transactions.db")  # Replace with actual DB
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df = load_data()

# Step 2: Data Preprocessing & Feature Engineering
df["transaction_hour"] = pd.to_datetime(df["transaction_time"]).dt.hour
df["is_high_amount"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
df["avg_customer_amount"] = df.groupby("customer_id")["amount"].transform("mean")
df["num_transactions"] = df.groupby("customer_id")["transaction_id"].transform("count")

features = ["amount", "transaction_hour", "is_high_amount", "avg_customer_amount", "num_transactions"]
X = df[features]
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train and Evaluate Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {results[name]:.4f}")

# Step 4: Visualization
sns.heatmap(confusion_matrix(y_test, models["RandomForest"].predict(X_test)), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - RandomForest")
plt.show()

# Step 5: Deploy as a Streamlit App with Real-Time Alerts
def fraud_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = models["RandomForest"].predict(input_df)[0]
    if prediction:
        send_alert(input_data)
    return int(prediction)

def send_alert(input_data):
    alert_data = {
        "message": "Potential fraud detected!",
        "transaction_details": input_data
    }
    requests.post("https://your-alert-api.com/notify", json=alert_data)
    
st.title("Fraud Detection System")
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
hour = st.slider("Transaction Hour", 0, 23, 12)
high_amount = 1 if amount > df["amount"].quantile(0.95) else 0
avg_customer_amount = st.number_input("Average Customer Transaction Amount", min_value=0.0, format="%.2f")
num_transactions = st.number_input("Number of Transactions by Customer", min_value=1, format="%d")

if st.button("Predict Fraud"):
    input_data = [amount, hour, high_amount, avg_customer_amount, num_transactions]
    prediction = fraud_prediction(input_data)
    st.write("Fraudulent Transaction" if prediction else "Legitimate Transaction")