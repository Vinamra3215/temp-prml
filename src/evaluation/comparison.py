"""
Master results table builder.
"""
import os
import pandas as pd


def append_result(results_dir, run_data):
    """Append a single experiment result to master CSV."""
    master_path = os.path.join(results_dir, "master.csv")
    df_new = pd.DataFrame([run_data])
    if os.path.exists(master_path):
        df_existing = pd.read_csv(master_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        os.makedirs(results_dir, exist_ok=True)
        df = df_new
    df.to_csv(master_path, index=False)
    return df


def load_master_results(results_dir):
    """Load the master results CSV."""
    master_path = os.path.join(results_dir, "master.csv")
    if not os.path.exists(master_path):
        return pd.DataFrame()
    return pd.read_csv(master_path)


def print_comparison_table(results_dir):
    """Print formatted comparison table."""
    df = load_master_results(results_dir)
    if df.empty:
        print("No results found.")
        return
    df_sorted = df.sort_values("test_accuracy", ascending=False)
    print(df_sorted.to_string(index=False))
