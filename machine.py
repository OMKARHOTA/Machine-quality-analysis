import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load all models
log_reg = pickle.load(open("logreg (1).pkl", "rb"))
dt = pickle.load(open("dt (1).pkl", "rb"))
kmeans = pickle.load(open("kmeans (1).pkl", "rb"))

st.title("ML Models Web App 🚀")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ("Logistic Regression", "Decision Tree", "KMeans")
)

# Sidebar checkbox for graph
show_graph = st.sidebar.checkbox("Show Model Graph")

# Inputs for Decision Tree / KMeans
f1 = st.number_input("Mean Process Temperature [K]")
f2 = st.number_input("Mean Air Temperature [K]")

# ✅ Use DataFrame with column names for consistency
data1 = pd.DataFrame([[f1, f2]], columns=["Mean Process Temperature [K]", "Mean Air Temperature [K]"])

# Inputs for Logistic Regression
f3 = st.number_input("Rotational speed [rpm]")
f4 = st.number_input("Process temperature [K]")

# ✅ Use DataFrame with correct feature names
# ✅ Match exactly with training column names
data2 = pd.DataFrame([[f3, f4]], columns=["Rotational speed [rpm]", "Process temperature [K]"])


# Prediction
if st.button("Predict"):
    if model_choice == "Decision Tree":
        pred = dt.predict(data1)
        st.success(f"Decision Tree Prediction: {pred[0]}")

    elif model_choice == "KMeans":
        pred = kmeans.predict(data1)
        st.success(f"KMeans Cluster: {pred[0]}")

    elif model_choice == "Logistic Regression":
        pred = log_reg.predict(data2)
        st.success(f"Logistic Regression Prediction: {pred[0]}")



