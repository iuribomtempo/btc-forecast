# 📊 BTC Forecast Project

This project aims to implement and compare different forecasting models for Bitcoin (BTC) price prediction, using machine learning and statistical approaches. The objective is to evaluate model performance and simulate trading strategies based on forecast signals.

## 📁 Project Structure

- `data/` – Raw and processed datasets
- `notebooks/` – Exploratory and modeling Jupyter notebooks
- `scripts/` – Python scripts for training and evaluation
- `models/` – Saved models (e.g., `.pkl`, `.h5`)
- `results/` – Forecast outputs, performance metrics, visualizations

## ⚙️ Installation

To install the required packages, first create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Note: This project uses Python 3.13.5.

🤖 Implemented Models
 ARIMA (AutoRegressive Integrated Moving Average)

 LSTM (Long Short-Term Memory)

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

RMSE – Root Mean Squared Error

MAE – Mean Absolute Error

MAPE – Mean Absolute Percentage Error

R² Score – Coefficient of Determination

🧠 Requirements
Dependencies are listed in requirements.txt. To export them from your virtual environment:

pip freeze > requirements.txt

👨‍💻 Author
Developed by Iuri Bomtempo for academic research and personal learning in Data Science and Forecasting.

Feel free to contribute or fork this project for your own research.