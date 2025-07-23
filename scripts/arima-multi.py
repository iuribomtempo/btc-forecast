import os
import pandas as pd
import numpy as np
from utils.arima_utils import (
    load_and_prepare,
    train_test_split,
    train_arima,
    static_arima_forecast,
    rolling_arima_forecast,
    reconstruct_prices_from_log_close,
    evaluate_model,
    plot_forecast_from_df,
    validate_ljung_box,
    validate_jarque_bera
)

# List of dataset file paths to run the pipeline on
DATASETS = [
    "data/raw/btcusdt_1d.csv",
    "data/raw/ethusdt_1d.csv",
]

# Set to True to run one model per year, or False to run only the whole dataset
SPLIT_BY_YEAR = True

def split_df_by_year(df):
    """
    Receives a DataFrame with a datetime index (or converts it if necessary),
    and returns a dict {year: DataFrame for that year}.
    """
    # Convert index to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    years = df.index.year.unique()
    dfs_by_year = {year: df[df.index.year == year].copy() for year in years}
    return dfs_by_year

# DataFrames to accumulate metrics and diagnostics for all models
all_models_metrics = []
all_models_diagnostics = []

# Main pipeline
for file_path in DATASETS:
    print(f"\n=== Running ARIMA for: {file_path} ===")

    symbol = os.path.splitext(os.path.basename(file_path))[0]

    # Load and prepare the dataset
    df = load_and_prepare(file_path)

    # Split into {year: DataFrame} if needed, otherwise just use the whole dataset
    if SPLIT_BY_YEAR:
        dfs_for_loop = split_df_by_year(df)
    else:
        dfs_for_loop = {None: df}  # single entry for the whole dataset

    for year, df_loop in dfs_for_loop.items():
        # Set up result directory and file suffix
        if year is not None:
            result_dir = os.path.join("results", "ARIMA", symbol, str(year))
            file_suffix = f"_{year}"
        else:
            result_dir = os.path.join("results", "ARIMA", symbol)
            file_suffix = ""

        os.makedirs(result_dir, exist_ok=True)

        # Create metrics and diagnostics DataFrames for this run
        metrics_df = pd.DataFrame(index=["MAPE", "RMSE", "R²"])
        diagnostics_df = pd.DataFrame(
            columns=['Model', 'Test', 'Statistic', 'p-value', 'Conclusion']
        )

        # Train/test split
        df_train, df_test, _ = train_test_split(df_loop, train_ratio=0.80, test_ratio=0.20)

        # Fit ARIMA (auto_arima)
        model = train_arima(df_train['log_close'], trace=False)

        # Save model summary for this dataset/year
        summary_path = os.path.join(result_dir, f"arima_model_summary{file_suffix}.txt")
        with open(summary_path, "w") as f:
            f.write(str(model.summary()))

        # Static forecast
        forecast_static_model_df = static_arima_forecast(model, df_test, n_periods=len(df_test))

        # Model evaluation (train and test)
        evaluate_model(
            y_true=df_train['close'].values,
            y_pred=np.exp(model.predict_in_sample()),
            column_name='ARIMA Static - Train',
            metrics_df=metrics_df
        )
        evaluate_model(
            y_true=forecast_static_model_df['real_price'],
            y_pred=forecast_static_model_df['predicted_price'],
            column_name='ARIMA Static - Test',
            metrics_df=metrics_df
        )

        # Diagnostics (STATIC)
        residuals_static = model.arima_res_.resid
        validate_ljung_box(residuals_static, lags=10, label='ARIMA Static (train)', diagnostics_df=diagnostics_df)
        validate_jarque_bera(residuals_static, label='ARIMA Static (train)', diagnostics_df=diagnostics_df)

        # Rolling forecast
        forecast_rolling_model_df = rolling_arima_forecast(df_train, df_test, order=model.order)
        last_price = df_train['close'].iloc[-1]
        forecast_rolling_model_df = reconstruct_prices_from_log_close(forecast_rolling_model_df, last_price)

        # Model evaluation (rolling)
        evaluate_model(
            forecast_rolling_model_df['real_price'],
            forecast_rolling_model_df['predicted_price'],
            column_name='ARIMA Rolling - Test',
            metrics_df=metrics_df
        )

        # Plots (full and last 50)
        plot_forecast_from_df(
            forecast_static_model_df,
            title=f"ARIMA Static Forecast - {symbol.upper()}{file_suffix}",
            save_path=os.path.join(result_dir, f"arima_static_price{file_suffix}.png"),
        )
        plot_forecast_from_df(
            forecast_rolling_model_df,
            title=f"ARIMA Rolling Forecast - {symbol.upper()}{file_suffix}",
            save_path=os.path.join(result_dir, f"arima_rolling_price{file_suffix}.png"),
        )
        plot_forecast_from_df(
            forecast_rolling_model_df[-50:],
            title=f"ARIMA Rolling Forecast (Last 50) - {symbol.upper()}{file_suffix}",
            save_path=os.path.join(result_dir, f"arima_rolling_price_last_50{file_suffix}.png"),
        )

        # Diagnostics (ROLLING)
        validate_ljung_box(forecast_rolling_model_df['residual'], lags=10, label='ARIMA Rolling (Test)', diagnostics_df=diagnostics_df)
        validate_jarque_bera(forecast_rolling_model_df['residual'], label='ARIMA Rolling (Test)', diagnostics_df=diagnostics_df)

        # Save results for this run
        forecast_static_model_df.to_csv(os.path.join(result_dir, f"arima_static_forecast{file_suffix}.csv"))
        forecast_rolling_model_df.to_csv(os.path.join(result_dir, f"arima_rolling_forecast{file_suffix}.csv"))
        metrics_df.to_csv(os.path.join(result_dir, f"metrics{file_suffix}.csv"))
        diagnostics_df.to_csv(os.path.join(result_dir, f"stats_diagnostics{file_suffix}.csv"))

        # Collect metrics and diagnostics for summary
        for col in metrics_df.columns:
            row = metrics_df[col].to_dict()
            row['dataset'] = symbol + (f"_{year}" if year is not None else "")
            row['split/model'] = col
            all_models_metrics.append(row)
        for _, row in diagnostics_df.iterrows():
            diag_row = row.to_dict()
            diag_row['dataset'] = symbol + (f"_{year}" if year is not None else "")
            all_models_diagnostics.append(diag_row)

        print(f"=== Finished for: {symbol}{file_suffix} ===\n")

# Save global metrics and diagnostics for comparison across all models
all_models_metrics_df = pd.DataFrame(all_models_metrics)
all_models_metrics_df = all_models_metrics_df[['dataset', 'split/model', 'MAPE', 'RMSE', 'R²']]
all_models_metrics_df.to_csv("results/ARIMA/all_models_metrics.csv", index=False)

all_models_diagnostics_df = pd.DataFrame(all_models_diagnostics)
cols = ['dataset', 'Model', 'Test', 'Statistic', 'p-value', 'Conclusion']
all_models_diagnostics_df = all_models_diagnostics_df[cols]
all_models_diagnostics_df.to_csv("results/ARIMA/all_models_diagnostics.csv", index=False)
