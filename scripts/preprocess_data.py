# -*- coding: utf-8 -*-

import os
import pandas as pd

def preprocess_csv_to_parquet(input_path, output_path):
    df = pd.read_csv(input_path)

    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated()]
    df = df.ffill().dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="fastparquet")
    print(f"[✓] Processado: {os.path.basename(input_path)} → {os.path.basename(output_path)}")

def process_all_csv_files(raw_dir="data/raw", processed_dir="data/processed"):
    for filename in os.listdir(raw_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(raw_dir, filename)
            output_filename = filename.replace(".csv", ".parquet")
            output_path = os.path.join(processed_dir, output_filename)
            preprocess_csv_to_parquet(input_path, output_path)

if __name__ == "__main__":
    process_all_csv_files()
