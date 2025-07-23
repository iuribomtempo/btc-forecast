# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:05:23 2025
@author: Iuri Bomtempo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera


# DF for error metrics
metrics_df = pd.DataFrame(index=["MAPE", "RMSE", "R²"])

# Df for Ljung-Box and Jarque-Bera tests
diagnostics_df = pd.DataFrame(
    columns=['Model', 'Test', 'Statistic', 'p-value', 'Conclusion']
    )


# Loads the CSV, converts date column, removes missing values, and adds log-transformed prices.
def load_and_prepare(file_path):
    """
    Loads a CSV file, parses the close_time column to datetime (date only),
    sorts by date, removes missing values, and sets 'close_time' as index.
    
    Applies logarithmic transformation to the 'close' price column.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file.

    Returns:
    --------
    df : pd.DataFrame
        Prepared DataFrame with 'close' prices and datetime index.

    """

    df = pd.read_csv(file_path)
    df['close_time'] = pd.to_datetime(df['close_time']).dt.date
    df = df.sort_values('close_time').reset_index(drop=True)
    df = df[['close_time', 'close']].dropna()
    df.set_index('close_time', inplace=True)
    df['log_close'] = np.log(df['close'])
    return df


# Performs Augmented Dickey-Fuller test to check if the time series is stationary.
def adf_test(series, name='Series'):
    """
    Performs the Augmented Dickey-Fuller test on a time series.

    Parameters:
    -----------
    series : pd.Series
        Time series to test.

    name : str
        Label to print in the output.

    Returns:
    --------
    None
    """
    result = adfuller(series.dropna())
    
    print(f"\nADF Test - {name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    
    if result[1] < 0.05:
        print("✅ Series is stationary")
    else:
        print("❌ Series is not stationary")
    

# Plots ACF and PACF side by side to inspect lags and autocorrelation patterns.
def plot_acf_pacf(series, lags=30, filename=''):
    """
    Plots ACF and PACF side-by-side for a given time series and saves as SVG.
    """

    font = {'family': 'Arial', 'size': 11}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm')

    axes[0].set_title('ACF', fontdict=font)
    axes[1].set_title('PACF', fontdict=font)

    for ax in axes:
        ax.tick_params(labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(filename, format='png')
    plt.show()


# Splits the time series DataFrame into train, test, and optional out-of-sample sets based on ratios.
def train_test_split(df, train_ratio, test_ratio, out_sample_ratio=0.0):
    """
    Splits the dataset into train, test, and out-of-sample sets.

    Parameters:
    -----------
    df : pd.DataFrame
        Complete dataset with datetime index.

    train_ratio : float
        Proportion of the dataset to be used for training.

    test_ratio : float
        Proportion to be used for testing.

    out_sample_ratio : float, optional (default=0.0)
        Proportion to be used for out-of-sample forecasting or hold-out.

    Returns:
    --------
    Tuple:
        - df_train : pd.DataFrame
        - df_test : pd.DataFrame
        - df_out_sample : pd.DataFrame or None
    """
    total_len = len(df)

    train_end = int(total_len * train_ratio)
    test_end = train_end + int(total_len * test_ratio)
    out_end = test_end + int(total_len * out_sample_ratio)

    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:test_end]

    if out_sample_ratio > 0.0:
        df_out_sample = df.iloc[test_end:out_end]
    else:
        df_out_sample = None


    return df_train, df_test, df_out_sample


# Fits an ARIMA model using auto_arima for optimal p, d, q selection with optional trace display.
def train_arima(series_train, trace=True):
    """
    Fits an ARIMA model to the training time series using auto_arima
    for automatic parameter selection.

    Parameters:
    ----------
    series_train : pd.Series
        Time series data to fit the ARIMA model on (typically log-transformed
                                                    price).

    trace : bool, default=True
        If True, displays the stepwise search process and chosen model.

    Returns:
    -------
    model : ARIMA object
        The fitted ARIMA model.
    """
    model = auto_arima(series_train, seasonal=False, trace=trace,
                       suppress_warnings=True, stepwise=True)
    print(model.summary())
    return model


# Uses a trained ARIMA model to forecast n steps ahead without retraining and returns price-level predictions.
def static_arima_forecast(model, df_test, n_periods):
    """
    Forecasts log(price) using a pre-fitted ARIMA model and transforms it back
    to original price scale.

    Parameters:
    ----------
    model : trained ARIMA model (e.g., from pmdarima)

    df_test : pd.DataFrame
        Test set with 'close' price column and datetime index.

    n_periods : int
        Number of future steps to forecast.

    Returns:
    -------
    forecast_df : pd.DataFrame with:
        - real_price: actual closing prices
        - predicted_price: forecasted prices
        - lower_price, upper_price: 95% confidence bounds
    """

    forecast_log, conf_int_log = model.predict(n_periods=n_periods,
                                               return_conf_int=True,
                                               alpha=0.05)

    # Convert back to price scale
    forecast = np.exp(forecast_log)
    conf_int = np.exp(conf_int_log)

    forecast_index = df_test.index[:n_periods]

    forecast_df = pd.DataFrame({
        'real_price': df_test['close'].iloc[:n_periods].values,
        'predicted_price': forecast,
        'lower_bound': conf_int[:, 0],
        'upper_bound': conf_int[:, 1]
    }, index=forecast_index)

    return forecast_df


# Performs a realistic rolling one-step-ahead forecast by refitting the ARIMA model at each step.
def rolling_arima_forecast(df_train, df_test, column='log_close', order=None):
    """
    Performs rolling forecast with one-step-ahead ARIMA predictions.

    Parameters:
        df_train (DataFrame): training data (indexed by date)
        df_test (DataFrame): testing data to forecast (indexed by date)
        column (str): column to use for modeling (e.g., 'log_close')
        order (tuple): ARIMA order (p,d,q)

    Returns:
        forecast_df (DataFrame): contains real, predicted, and confidence bounds
    """
    
    preds = []
    lower_bounds = []
    upper_bounds = []
    reals = []
    residuals = []

    history = df_train[column].copy()

    for t in df_test.index:
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit()

            forecast_result = model_fit.get_forecast(steps=1)
            mean = forecast_result.predicted_mean.iloc[0]
            conf_int = forecast_result.conf_int().iloc[0]
            
            ###########################################################
            real_value = df_test.loc[t, column]
            residual = real_value - mean

            # 🖨️ Print para visualização
            print(f"\n📅 Date: {t}")
            print(f"🔮 Predicted (log): {mean:.5f}")
            print(f"✅ Real (log):      {real_value:.5f}")
            print(f"📉 95% CI:          [{conf_int[0]:.5f}, {conf_int[1]:.5f}]")
            ############################################################    

            preds.append(mean)
            lower_bounds.append(conf_int[0])
            upper_bounds.append(conf_int[1])
            reals.append(df_test.loc[t, column])
            residuals.append(residual)
           
            # Update the history with the real value of t day
            history = pd.concat([history, pd.Series(df_test.loc[t, column],
                                                    index=[t])])
        
        except Exception as e:
            ########################
            print(f"⚠️ Error on {t}: {e}")
            ########################
            print(f"Erro em {t}: {e}")
            preds.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            reals.append(df_test.loc[t, column])

    forecast_df = pd.DataFrame({
        'log_real': reals,
        'log_predicted': preds,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds,
        'residual': residuals
    }, index=df_test.index)

    return forecast_df


# Converts log-scale forecast results back to real price scale (exponential transformation).
def reconstruct_prices_from_log_close(forecast_df, last_price):
    """
    Reconstructs real prices from log forecasts.

    Parameters:
        forecast_df: contains 'real_price', 'predicted_price', 'lower_bound',
        'upper_bound' (log scale)
        last_train_price: last actual closing price from training set

    Returns:
        forecast_df: updated with prices in USDT
    """
    real_prices = []
    predicted_prices = []
    lower_prices = []
    upper_prices = []

    for i in range(len(forecast_df)):
        real_prices.append(np.exp(forecast_df['log_real'].iloc[i]))
        predicted_prices.append(np.exp(forecast_df['log_predicted'].iloc[i]))
        lower_prices.append(np.exp(forecast_df['lower_bound'].iloc[i]))
        upper_prices.append(np.exp(forecast_df['upper_bound'].iloc[i]))

    forecast_df['real_price'] = real_prices
    forecast_df['predicted_price'] = predicted_prices
    forecast_df['lower_bound'] = lower_prices
    forecast_df['upper_bound'] = upper_prices
    forecast_df['residual'] = forecast_df['real_price'] - forecast_df['predicted_price']
    
    # Remove log columns
    forecast_df.drop(columns=['log_real', 'log_predicted'], inplace=True)
    
    return forecast_df


# Computes and returns MAPE, RMSE, and R² for evaluating prediction accuracy.
def evaluate_model(y_true, y_pred, column_name):
    """
    Computes MAPE, RMSE, and R² for model evaluation.
    
    """
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Create a pd Series with the rsults
    metrics_series = pd.Series([mape, rmse, r2], index=["MAPE", "RMSE", "R²"], name=column_name)
    metrics_df[column_name] = metrics_series
    
    print("\n📊 Model Evaluation:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    return mape, rmse, r2


# Plots actual vs predicted prices with confidence intervals using the unified forecast DataFrame.
def plot_forecast_from_df(reconstructed_df, title='', save_path=None):
    """
    Plots ARIMA forecast with 95% confidence interval using a unified DataFrame
    that contains the following columns:
        - real_price
        - predicted_price
        - lower_price
        - upper_price

    Parameters:
    ----------
    reconstructed_df : pd.DataFrame
        DataFrame with forecasted and real prices in original (non-log) scale.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(reconstructed_df.index, reconstructed_df['real_price'], label='Real Price', color='black')
    plt.plot(reconstructed_df.index, reconstructed_df['predicted_price'], label='Predicted (ARIMA)', color='dodgerblue')
    plt.fill_between(reconstructed_df.index,
                     reconstructed_df['lower_bound'],
                     reconstructed_df['upper_bound'],
                     color='lightblue', alpha=0.4, label='95% Confidence Interval')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USDT)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png')
    
    plt.show()

# Applies Ljung-Box test to verify residuals are white noise (i.e., no autocorrelation).
def validate_ljung_box(residuals, lags=10, label=None):
    """
    Runs Ljung-Box test to verify if residuals are white noise
    (no autocorrelation). Appends result to diagnostics_df.

    Parameters:
        residuals (array-like): Model residuals (errors).
        lags (int): Number of lags to include in the test.
        label (str): Label for the model to appear in diagnostics_df.
    """
    ljung = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    stat = ljung['lb_stat'].values[0]
    p_value = ljung['lb_pvalue'].values[0]
    conclusion = "White noise" if p_value > 0.05 else "Autocorrelated"

    diagnostics_df.loc[len(diagnostics_df)] = [label, "Ljung-Box", stat, p_value, conclusion]

    print(f"\n🧪 Ljung-Box Test ({label}, lags={lags}):")
    print(ljung, "\n")

    if p_value > 0.05:
        print(f"✅ p-value = {p_value:.4f} > 0.05 → Fail to reject H₀.")
        print("✔️ Residuals behave as white noise. ARIMA model is well-fitted.")
    else:
        print(f"❌ p-value = {p_value:.4f} ≤ 0.05 → Reject H₀.")
        print("⚠️ Residuals show autocorrelation. Model might be misspecified.")


# Applies Jarque-Bera test to assess normality of model residuals.
def validate_jarque_bera(residuals, label=None):
    """
    Applies the Jarque-Bera test to determine if residuals follow a normal
    distribution. Appends result to diagnostics_df.

    Parameters:
        residuals (array-like): Model residuals (errors).
        label (str): Label for the model to appear in diagnostics_df.
    """
    jb_stat, p_value = jarque_bera(residuals)
    conclusion = "Normal" if p_value > 0.05 else "Non-normal"

    diagnostics_df.loc[len(diagnostics_df)] = [label, "Jarque-Bera", jb_stat, p_value, conclusion]

    print(f"\n🧪 Jarque-Bera Test ({label}):")
    print(f"JB Statistic: {jb_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value > 0.05:
        print("✅ p-value > 0.05 → Fail to reject H₀.")
        print("✔️ Residuals follow a normal distribution.")
    else:
        print("❌ p-value ≤ 0.05 → Reject H₀.")
        print("⚠️ Residuals do not follow a normal distribution.")