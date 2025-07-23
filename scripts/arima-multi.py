import os
import pandas as pd
from scripts.arima import (
    load_and_prepare,
    train_test_split,
    train_arima,
    static_arima_forecast,
    rolling_arima_forecast,
    reconstruct_prices_from_log_close,
    evaluate_model,
    plot_forecast_from_df,
)

#%%
# Lista de arquivos/datasets para rodar o pipeline
DATASETS = [
    "data/raw/btcusdt_1d.csv",
    "data/raw/btcusdt_1w.csv",
    "data/raw/ethusdt_1w.csv",
    # Adicione outros arquivos .csv conforme desejar
]

# Para cada dataset...
for file_path in DATASETS:
    print(f"\n=== Executando ARIMA para: {file_path} ===")

    # Nome para organizar resultados (ex: 'btcusdt_1d')
    symbol = os.path.splitext(os.path.basename(file_path))[0]
    result_dir = os.path.join("results", "ARIMA", symbol)
    os.makedirs(result_dir, exist_ok=True)

    # DataFrame para métricas e diagnósticos
    metrics_df = pd.DataFrame(index=["MAPE", "RMSE", "R²"])
    # Diagnostics
    diagnostics_df = pd.DataFrame(
        columns=['Model', 'Test', 'Statistic', 'p-value', 'Conclusion']
    )

    # 1. Carregar e preparar os dados
    df = load_and_prepare(file_path)

    # 2. Split em treino/teste
    df_train, df_test, _ = train_test_split(df, train_ratio=0.80, test_ratio=0.20)

    # 3. Treinar ARIMA (auto_arima)
    model = train_arima(df_train['log_close'], trace=False)

    # 4. Static Forecast
    forecast_static_model_df = static_arima_forecast(model, df_test, n_periods=len(df_test))

    # 5. Avaliar no treino e teste
    evaluate_model(
        y_true=df_train['close'].values,
        y_pred=np.exp(model.predict_in_sample()),
        column_name='ARIMA Static - Train'
    )
    evaluate_model(
        y_true=forecast_static_model_df['real_price'],
        y_pred=forecast_static_model_df['predicted_price'],
        column_name='ARIMA Static - Test'
    )

    # 6. Rolling Forecast
    forecast_rolling_model_df = rolling_arima_forecast(df_train, df_test, order=model.order)

    # 7. Reconstroi preços reais
    last_price = df_train['close'].iloc[-1]
    forecast_rolling_model_df = reconstruct_prices_from_log_close(
        forecast_rolling_model_df, last_price
    )

    # 8. Avaliação rolling
    evaluate_model(
        forecast_rolling_model_df['real_price'],
        forecast_rolling_model_df['predicted_price'],
        column_name='ARIMA Rolling - Test'
    )

    # 9. Plots (salva gráficos)
    plot_forecast_from_df(
        forecast_static_model_df,
        title=f"ARIMA Static Forecast - {symbol.upper()}",
        save_path=os.path.join(result_dir, "arima_static_price.png"),
    )
    plot_forecast_from_df(
        forecast_rolling_model_df,
        title=f"ARIMA Rolling Forecast - {symbol.upper()}",
        save_path=os.path.join(result_dir, "arima_rolling_price.png"),
    )

    # 10. Salvar resultados em CSV
    forecast_static_model_df.to_csv(os.path.join(result_dir, "arima_static_forecast.csv"))
    forecast_rolling_model_df.to_csv(os.path.join(result_dir, "arima_rolling_forecast.csv"))
    metrics_df.to_csv(os.path.join(result_dir, "metrics.csv"))
    diagnostics_df.to_csv(os.path.join(result_dir, "stats_diagnostics.csv"))

    print(f"=== Finalizado para: {symbol} ===\n")

print("Pipeline multi-ARIMA concluído!")
