import os
import pandas as pd
import matplotlib.pyplot as plt

RAW_PREFIX = "../../TSB/raw/TSB-U"
PROCESSED_PREFIX = "../../TSB/processed/TSB-U"

def plot_timeseries(df: pd.DataFrame, title: str, has_labels: bool = True):
    """
    Plot time series.
    If has_labels is True, anomalies (label==1) are shown in red, normal points in blue.
    If has_labels is False, the entire series is plotted in blue.
    """
    # X-axis: use timestamp if available, else index
    if "timestamp" in df.columns:
        x = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        x = df.index

    y = df["Data"]

    plt.figure(figsize=(12, 6))

    if has_labels and "Label" in df.columns:
        labels = df["Label"]
        normal = labels == 0
        anomalies = labels == 1
        plt.scatter(x[normal], y[normal], color="blue", s=10, label="Normal")
        plt.scatter(x[anomalies], y[anomalies], color="red", s=25, label="Anomaly")
        plt.plot(x, y, color="lightgray", linewidth=1, alpha=0.7)
    else:
        plt.plot(x, y, color="blue", linewidth=1.5, label="Processed Data")

    plt.title(title)
    plt.xlabel("Time" if "timestamp" in df.columns else "Index")
    plt.ylabel("Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot{has_labels}.png")

def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if "Data" not in df.columns:
        raise ValueError("CSV must contain 'value' column")
    return df


def plot_raw_and_processed(filename: str):
    """
    Plot both raw and processed data for a given filename.
    """
    raw_path = os.path.join(RAW_PREFIX, filename)
    processed_path = os.path.join(PROCESSED_PREFIX, filename)

    df_raw = load_csv(raw_path)
    df_processed = load_csv(processed_path)

    # Raw data has anomalies
    plot_timeseries(df_raw, f"Raw Data with Anomalies\n{filename}", has_labels=True)

    # Processed data has no anomalies/label
    plot_timeseries(df_processed, f"Processed Data\n{filename}", has_labels=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python plot_timeseries.py filename.csv")
        sys.exit(1)

    file_name = sys.argv[1]
    plot_raw_and_processed(file_name)
