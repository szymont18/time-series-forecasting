import pandas as pd


def load_data_from_file(file_path):
    df = pd.read_csv(file_path)
    series = pd.Series(df["Data"].values)
    return series