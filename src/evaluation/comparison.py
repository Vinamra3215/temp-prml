"""
Master results table builder — supports multiple output CSVs.
"""
import os
import pandas as pd


def append_result(results_dir, run_data, csv_name="master.csv"):
    """Append a single experiment result to a CSV file."""
    master_path = os.path.join(results_dir, csv_name)
    df_new = pd.DataFrame([run_data])
    if os.path.exists(master_path):
        df_existing = pd.read_csv(master_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        os.makedirs(results_dir, exist_ok=True)
        df = df_new
    df.to_csv(master_path, index=False)
    return df


def load_master_results(results_dir, csv_name="master.csv"):
    """Load a results CSV."""
    master_path = os.path.join(results_dir, csv_name)
    if not os.path.exists(master_path):
        return pd.DataFrame()
    return pd.read_csv(master_path)


def print_comparison_table(results_dir, csv_name="master.csv"):
    """Print formatted comparison table."""
    df = load_master_results(results_dir, csv_name)
    if df.empty:
        print("No results found.")
        return
    df_sorted = df.sort_values("test_accuracy", ascending=False)
    print(df_sorted.to_string(index=False))
