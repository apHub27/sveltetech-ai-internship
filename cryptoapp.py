import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import sys

st.set_page_config(page_title="Bitcoin Price Predictor")
st.title(" Bitcoin Price Predictor (RNN vs LSTM vs GRU)")
st.write("Predicts next-day Bitcoin price using the last 60 days of data.")
st.write("---")

# --- Model class definitions (needed to load the saved models) ---
class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class SimpleGRU(nn.Module):
    def __init__(self, hidden_size=50):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

sys.modules["__main__"].SimpleRNN = SimpleRNN
sys.modules["__main__"].SimpleLSTM = SimpleLSTM
sys.modules["__main__"].SimpleGRU = SimpleGRU

# --- Load models and scaler ---
@st.cache_resource
def load_everything():
    rnn = SimpleRNN()
    rnn.load_state_dict(torch.load("rnn_model.pth", map_location="cpu"))
    
    lstm = SimpleLSTM()
    lstm.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
    
    gru = SimpleGRU()
    gru.load_state_dict(torch.load("gru_model.pth", map_location="cpu"))
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    rnn.eval(); lstm.eval(); gru.eval()
    return rnn, lstm, gru, scaler

try:
    rnn_model, lstm_model, gru_model, scaler = load_everything()
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Fetch live recent data ---
st.subheader("Step 1: Fetch recent Bitcoin data")
if st.button("Fetch last 60 days of Bitcoin prices"):
    with st.spinner("Fetching live data..."):
        recent_data = yf.download("BTC-USD", period="70d")["Close"].values.reshape(-1, 1)
        recent_data = recent_data[-60:]  # exactly 60 days
        st.session_state["recent_data"] = recent_data
        st.line_chart(recent_data.flatten())
        st.write(f"Latest price: ${recent_data[-1][0]:,.2f}")

# --- Predict ---
st.subheader("Step 2: Predict next-day price")
model_choice = st.selectbox("Choose a model", ["RNN", "LSTM", "GRU", "Compare All"])

if st.button("Predict Next Day Price"):
    if "recent_data" not in st.session_state:
        st.warning("Please fetch recent data first (Step 1).")
    else:
        recent_data = st.session_state["recent_data"]
        scaled_input = scaler.transform(recent_data)
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)  # shape (1, 60, 1)

        def predict(model):
            with torch.no_grad():
                pred_scaled = model(input_tensor).numpy()
            return scaler.inverse_transform(pred_scaled)[0][0]

        if model_choice == "Compare All":
            rnn_pred = predict(rnn_model)
            lstm_pred = predict(lstm_model)
            gru_pred = predict(gru_model)
            st.write(f"**RNN Prediction:** ${rnn_pred:,.2f}")
            st.write(f"**LSTM Prediction:** ${lstm_pred:,.2f}")
            st.write(f"**GRU Prediction:** ${gru_pred:,.2f}")

            fig, ax = plt.subplots()
            ax.plot(recent_data.flatten(), label="Last 60 days (actual)")
            ax.axhline(rnn_pred, color="blue", linestyle="--", label="RNN Prediction")
            ax.axhline(lstm_pred, color="orange", linestyle="--", label="LSTM Prediction")
            ax.axhline(gru_pred, color="green", linestyle="--", label="GRU Prediction")
            ax.legend()
            st.pyplot(fig)
        else:
            model_map = {"RNN": rnn_model, "LSTM": lstm_model, "GRU": gru_model}
            pred = predict(model_map[model_choice])
            st.success(f"### {model_choice} Predicted Next-Day Price: ${pred:,.2f}")
