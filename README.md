# 📊 BTC Forecast Project

This project aims to implement and compare different forecasting models for Bitcoin (BTC) price prediction, using machine learning and statistical approaches. The objective is to evaluate model performance and simulate trading strategies based on forecast signals.

# 🤖 Implemented Models
 -ARIMA (AutoRegressive Integrated Moving Average)
 -LSTM (Long Short-Term Memory)

## 📁 Project Structure

- data/
  - raw/               # Raw data (e.g., OHLCV from Binance)
  - processed/         # Cleaned and transformed datasets

- results/
  - ARIMA/
    - *.png            # ACF/PACF and forecast plots (static and rolling)
    - *.csv            # Forecasts, performance metrics (MAPE, RMSE, R²), and diagnostic tests (Jarque-Bera and Ljung-Box)
    - arima_model_summary.txt  # Summary of ARIMA model parameters

- scripts/
  - get_crypto_data.py     # Downloads OHLCV data from Binance
  - preprocess_data.py     # Data cleaning, log transformation, and preparation
  - arima.py               # Static and rolling ARIMA implementation (single dataset)
  - lstm.py                # LSTM model implementation (single dataset)

- notebooks/               # Jupyter notebooks for exploration and modeling
- models/                  # Saved models (.pkl, .h5, etc.)
- requirements.txt         # Project dependencies
- README.md                # Project description and instructions
- venv/                    # Virtual environment (not versioned)

## ⚙️ Installation

1. **Create and activate a virtual environment (Python 3.11.9)**:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt # On Windows
# source venv\Scripts\activate # On macOS/Linux

Note: This project uses PPython 3.11.9.

🚀 Usage
You can either run the scripts directly or explore the notebooks interactively.

Running scripts
python scripts/ARIMA.py
python scripts/LSTM.py
Use Spyder or Jupyter to open notebooks:

bash
notebooks/arima_model.ipynb
notebooks/lstm_model.ipynb

📈 Evaluation Metrics
Forecast accuracy is evaluated using the following metrics:

RMSE - Root Mean Squared Error

MAE - Mean Absolute Error

MAPE - Mean Absolute Percentage Error

R² Score - Coefficient of Determination

🧠 Requirements
Dependencies are listed in requirements.txt. To export them from your virtual environment:

pip freeze > requirements.txt

👨‍💻 Author
Developed by Iuri Bomtempo for academic research and personal learning in Data Science and Forecasting.

Feel free to contribute or fork this project for your own research.