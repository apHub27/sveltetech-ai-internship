
import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="SvelteTech Guardrail", page_icon="🛡️")
st.title("🛡️ SvelteTech RAG Prompt Guardrail Portal")
st.write("---")

# Load saved model  file
try:
    with open('guardrail_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'guardrail_model.pkl' not found. Please train the model first.")

# Create the input fields
user_input = st.text_input("Enter Customer Message:", "Type here...")

if st.button("Analyze Prompt Securely"):
    length = len(user_input)
    danger_keywords = ["ignore", "admin", "bypass", "secrets", "attack", "override"]
    danger_count = sum(1 for word in danger_keywords if word in user_input.lower())
    
    # Process the feature vector
    feature_vector = np.array([[length, danger_count]])
    prediction = loaded_model.predict(feature_vector)
    
    # Render interactive dashboard alerts
    if prediction == 1:
        st.error("🚨 ALERT: Malicious Prompt Attack Detected! Route blocked completely.")
    else:
        st.success("✅ SAFE: Normal customer query. Routing to cheap database.")
