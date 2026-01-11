# 📈 Tesla Stock Price Prediction using LSTM

An end-to-end **time-series forecasting project** that predicts Tesla’s stock **Adjusted Closing Price** for the next **1, 5, and 10 days** using historical market data. The project focuses on building a **production-ready ML pipeline**, covering data analysis, feature engineering, deep learning modeling, and deployment via **Streamlit Cloud**.

---

## 🚀 Live Demo

🔗 **Streamlit App:** *https://tsla-stock-price-prediction.streamlit.app*

---

## 🧠 Project Overview

Stock prices are sequential in nature and exhibit temporal dependencies. Traditional ML models struggle to capture these patterns effectively. This project leverages a **Long Short-Term Memory (LSTM)** neural network to model historical dependencies and perform **multi-horizon forecasting**.

### Forecasting Horizons

* 📅 **1 Day Ahead**
* 📅 **5 Days Ahead**
* 📅 **10 Days Ahead**

---

## 🗂 Dataset

The dataset contains historical Tesla stock data with the following columns:

* `Date`
* `Open`
* `High`
* `Low`
* `Close`
* `Adj Close`
* `Volume`

Source: Publicly available stock market data

---

## 🔧 Feature Engineering

To help the model learn meaningful patterns, several time-series specific features were created:

* **Returns**: Percentage change in adjusted closing price
* **Moving Averages**: 10-day and 20-day rolling means
* **Volatility**: 10-day rolling standard deviation

These features help stabilize the learning process and provide trend and momentum information to the model.

---

## 🏗 Model Architecture

* **Model Type**: Stacked LSTM
* **Input**: Sliding window of past 60 days
* **Output**: Multi-output regression (1, 5, 10 day forecasts)
* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Adam

The model predicts all future horizons **simultaneously**, enabling consistent multi-step forecasting.

---

## 🧪 Evaluation

Model performance is evaluated using standard regression metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Predictions are inverse-scaled to obtain results in actual price units.

---

## 🌐 Deployment (Streamlit)

The trained model is deployed using **Streamlit**, with special care taken to ensure cloud compatibility:

* Model and scalers are cached for efficient inference
* No retraining during runtime
* Lightweight inference pipeline
* CPU-only execution (Streamlit Cloud friendly)

Users can generate real-time predictions for upcoming days based on the latest available data.

---

## 📁 Project Structure

```
TSLA_stock_price_prediction/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── README.md
│
├── data/
│   └── processed_data/
|       ├── feat_engg.csv
│       └── TSLA_date_converted.csv
|   └── raw_data/
|       └── TSLA.csv
│
├── models/
│   ├── lstm_tsla.h5
│   └── scalers/
|       ├── feature_scaler.joblib
│       └── target_scaler.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engg.ipynb
│
└── src/
    ├── preprocessing.py
    ├── sequence_generator.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Rupeshyadv/TSLA-stock-price-prediction.git
cd TSLA-stock-price-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 📌 Key Learnings

* Importance of **time-series aware feature engineering**
* Preventing **data leakage** during scaling and sequence generation
* Designing models for **multi-horizon forecasting**
* Building ML systems with **deployment constraints** in mind

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does **not** constitute financial or investment advice.

---

## 🙌 Acknowledgements

* TensorFlow / Keras
* Streamlit
* Pandas, NumPy, Scikit-learn

---

⭐ If you found this project helpful, feel free to star the repository!
