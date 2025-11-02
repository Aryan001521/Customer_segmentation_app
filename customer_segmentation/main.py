import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib 
import streamlit as st 

# Load saved models
kmeans = joblib.load('customer_segmentation/kmeans_models.pkl')
scaler = joblib.load('customer_segmentation/scaler.pkl')

# --- Custom CSS for background and styling ---
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    /* Center the title */
    h1 {
        text-align: center;
        color: #00FFFF;
        font-weight: 800;
    }

    /* Number inputs transparent */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white !important;
        border-radius: 10px;
        border: 1px solid #00FFFF;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #00FFFF;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1affd5;
        transform: scale(1.05);
    }

    /* Markdown table style */
    table {
        width: 100%;
        border-collapse: collapse;
        background-color: rgba(255,255,255,0.1);
        color: white;
        border-radius: 12px;
        overflow: hidden;
    }
    th, td {
        border: 1px solid #00FFFF;
        padding: 10px;
        text-align: left;
    }
    th {
        background-color: rgba(0,255,255,0.3);
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🧠 Customer Segmentation Dashboard")

st.write("Fill in customer details to predict their segment 👇")

# --- Input Fields ---
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
Total_spending = st.number_input("Total Spending (Sum of Purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visit = st.number_input("Number of Web Visits (per Month)", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# Prepare input data
input_data = pd.DataFrame({
    "Age": [Age],
    "Income": [Income],
    "Total_Spending": [Total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visit],
    "Recency": [recency]
})

# Scale input
input_scaled = scaler.transform(input_data)

# --- Prediction Button ---
if st.button("🔍 Predict Customer Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"✅ Predicted Segment: **Cluster {cluster}**")

    # Cluster info dictionary
    cluster_info = {
        0: {
            "Interpretation": "💎 Premium Loyal Customers",
            "Reasoning": "High income (~76k), high spending (1300+), frequent both web & store purchases, low recency → very active, high-value"
        },
        1: {
            "Interpretation": "🛍️ Mid-Income Regular Shoppers",
            "Reasoning": "Moderate income (~34k), low spending (~120), average recency → shop occasionally, stable segment"
        },
        2: {
            "Interpretation": "🌐 Digital Buyers",
            "Reasoning": "High web purchases, low store purchases, high income (~72k), high spending (~1138), very active online"
        },
        3: {
            "Interpretation": "💤 Dormant Customers",
            "Reasoning": "Low income (~36k), low spending (~135), very high recency (~75) → inactive customers, likely churned"
        },
        4: {
            "Interpretation": "💻 Omnichannel Shoppers",
            "Reasoning": "Mid-high income (~59k), good spending (~875), very frequent web & store purchases, moderate recency → balanced and loyal"
        },
        5: {
            "Interpretation": "💸 Budget Customers",
            "Reasoning": "Low income (~66k but possibly data skew), very low spending (~62), minimal activity both online & offline"
        }
    }

    # Display cluster info in a table
    info = cluster_info.get(cluster, None)
    if info:
        st.markdown(f"""
        | **Cluster** | **Interpretation** | **Reasoning** |
        |--------------|--------------------|----------------|
        | {cluster} | {info['Interpretation']} | {info['Reasoning']} |
        """, unsafe_allow_html=True)
    else:
        st.warning("Cluster information not found.")

