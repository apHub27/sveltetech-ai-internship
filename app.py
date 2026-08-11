import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="SvelteTech Guardrail", page_icon="🛡️")

st.title("🛡️  RAG Prompt Guardrail Portal")

# loading model file
try:
    with open('guardrail_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    pass

user_input = st.text_input("Enter Customer Message:", "Type here...")

if st.button("Analyze"):
    #  Hardcore Rules Sheet (Security Layer)
    danger_keywords = ["ignore", "admin", "bypass", "secrets", "attack", "override", "rules", "access"]
    
    # Check if any danger keyword exists in user input
    user_input_lower = user_input.lower()
    has_danger_word = any(word in user_input_lower for word in danger_keywords)
   
    # Agar message me koi bhi danger word mil gaya, toh direct block (1)
    if has_danger_word:
        st.error("🚨 ALERT: Malicious Prompt Attack Detected! Route blocked completely.")
    else:
        st.success("✅ SAFE: Normal customer query. Routing to cheap database.")
