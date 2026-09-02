# 🛡️ SvelteTech AI Internship Portfolio

Welcome to my 20-day machine learning and AI core engineering log.

---

## 🚀 Featured Project: RAG Prompt Guardrail System (Day 10)
An end-to-end security pipeline that performs custom feature engineering on text queries to block malicious prompt injections using a Random Forest Ensemble model.

*   **💻 Live Interactive App:** ((https://sveltetech-ai-internship-tpantcjetae6kupkfu6llh.streamlit.app/))

### 📺 System Execution Demo:
![RAG Guardrail App Demo](./Video_Project.gif)

---

House price predictor with Random Forest and location dropdown UI
deployed: https://8501-m-s-kkb-use1c1-18megw66scakm-c.us-east1-1.prod.colab.dev/

## 📈 Ongoing Curriculum Progress Track
*   **Day 1-2:** Probability & Descriptive Statistics Baseline
*   **Day 3-5:** Feature Engineering, Imputation, & Train-Test Splitting
*   **Day 6-8:** Linear vs Logistic Regression Baselines
*   **Day 9-10:** Decision Trees & Random Forests
*   **Day 11:** Support Vector Machines (SVM) with Real Web Data
---

Bitcoin Price Prediction — RNN vs LSTM vs GRU

A time-series deep learning project that predicts next-day Bitcoin price using three different recurrent neural network architectures — RNN, LSTM, and GRU — and compares their performance fairly.

 Live App: [https://sveltetech-ai-internship-qbbrjnia2f4z6yshgz8avv.streamlit.app/]

📌 Project Overview

This project answers: can we predict Bitcoin's next-day price from its past 60 days of prices, and which recurrent architecture (RNN, LSTM, or GRU) performs best at this task?

Rather than just picking one architecture, all three were trained on identical data with identical settings, so the comparison is fair — architecture is the only variable that changes.

📊 Data Source
Data: Bitcoin (BTC-USD) daily closing prices, 2019–2024, pulled live via the yfinance library
Size: ~2,191 days of price data
Target: Next-day closing price, predicted from the previous 60 days


⚙️ How It Works — Pipeline Overview
Raw price data (yfinance)
   → Scale prices to 0–1 range (MinMaxScaler)
   → Build 60-day sequences (past 60 days → predict day 61)
   → Chronological train/test split (NO shuffling — time series data)
   → Train RNN, LSTM, and GRU (equal epochs, same data)
   → Evaluate on test set (MAE, RMSE, % error)
   → Deploy as a Streamlit app for live predictions

  1. Preprocessing
Prices scaled to a 0–1 range using MinMaxScaler, since neural networks train far better on small, consistent-range numbers than raw prices ranging from $3,800 to $106,000
Data converted into sequences: each training example is "60 days of prices → the next day's price"
Train/test split is NOT shuffled — unlike a typical ML split, shuffling time-series data would let the model "see the future" during training. The test set is strictly the most recent ~20% of the timeline.
2. Model Training

Three architectures were trained on identical sequences:

RNN — simplest recurrent architecture, but prone to the vanishing gradient problem over longer sequences
LSTM — adds a gated memory cell (Forget/Input/Output gates) to preserve information over longer sequences
GRU — a simplified LSTM variant with fewer gates (Reset/Update), often faster to train
3. A Key Finding — Fair Comparison Matters

An initial comparison at 50 epochs made GRU look far worse (47% error) than LSTM (11% error). Investigating further revealed GRU's training loss was still dropping rapidly — it simply hadn't finished converging yet. All three models were retrained for 100 epochs to ensure a fair comparison.

Once properly trained:

Model	MAE	% Error
GRU	$2,136.56	3.4% (best)
LSTM	$3,256.08	5.2%
RNN	$3,313.48	5.3%

Lesson: an apparent "winner" after limited training can be misleading — training time must be controlled for before drawing architectural conclusions.

4. Limitations Observed

All three models show noticeable prediction lag during sharp, sudden price movements (e.g., Bitcoin's rapid rally toward the end of the test period). This is a common characteristic of sequence models — they learn from historical patterns, and sudden trend breaks aren't well represented in past data windows.

5. Deployment
Trained models saved via torch.save() and loaded in a Streamlit app using state_dict loading (loading weights into a freshly instantiated model class, rather than pickling the full model object)
The app fetches live recent Bitcoin data via yfinance, lets the user choose a model (or compare all three), and displays the predicted next-day price alongside a chart
🛠️ Tech Stack
Language: Python
Deep Learning: PyTorch (nn.RNN, nn.LSTM, nn.GRU)
Data: yfinance, pandas, numpy
Preprocessing: scikit-learn (MinMaxScaler)
UI: Streamlit
Deployment: Streamlit Community Cloud
Visualization: Matplotlib
📁 Project Files
cryptoapp.py          - Streamlit web app
rnn_model.pth          - Trained RNN weights (state_dict)
lstm_model.pth         - Trained LSTM weights (state_dict)
gru_model.pth          - Trained GRU weights (state_dict)
scaler.pkl             - Fitted MinMaxScaler (for consistent input scaling)
rnn_project.ipynb       - Full training notebook
requirements.txt        - Python dependencies
README.md

⚠️ Known Issue Fixed During Deployment

Model files were initially saved and loaded as full pickled objects (torch.save(model, ...)), but ended up containing only the model's weights (state_dict) rather than the complete model object — causing a load failure ('OrderedDict' object has no attribute 'eval') on deployment. Fixed by instantiating the model architecture first, then loading the saved weights into it via load_state_dict() — a more robust pattern than pickling full model objects.



👤 Author
Ankit — AI Intern, SvelteTech

👤 Author
Ankit — AI Intern, SvelteTech
