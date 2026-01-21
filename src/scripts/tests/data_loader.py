import pandas as pd
import importlib
import os
from sktime.split import temporal_train_test_split

def load_and_split(config):
    data = None

    if config.get("source") == "sktime":
        try:
            module = importlib.import_module("sktime.datasets")
            loader = getattr(module, config["loader"])
            data = loader()
        except Exception as e:
            raise ValueError(f"Could not load sktime data {config['name']}: {e}")

    elif config.get("source") == "file":
        path = config["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            df = pd.read_csv(path)

            date_col = None
            possible_date_cols = ['date', 'ds', 'timestamp', 'Date', 'Time']
            for col in df.columns:
                if col in possible_date_cols:
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            else:
                df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="h")

            if "Data" in df.columns:
                data = df[["Data"]]
            elif "y" in df.columns:
                data = df[["y"]]
            else:
                data = df

        except Exception as e:
            raise ValueError(f"Error reading file {path}: {e}")

    if data is None:
        raise ValueError(f"Could not load data for {config['name']}")

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range(start="2000-01-01", periods=len(data), freq="h")

        if "dim_0" in data.columns and len(data.columns) == 1:
            data = data.astype(float)
            data.columns = ['y']

    if "preprocess" in config and config["preprocess"] is not None:
        preprocess_func = config["preprocess"]
        data = preprocess_func(data)

    if config["type"] == "U":
        MAX_HISTORY_LIMIT = 1500
    else:
        MAX_HISTORY_LIMIT = 900

    if len(data) > MAX_HISTORY_LIMIT:
        print(f"[INFO] Dataset too large ({len(data)}). Truncating to last {MAX_HISTORY_LIMIT} points.")
        data = data.iloc[-MAX_HISTORY_LIMIT:]

    y_train_full, y_test = temporal_train_test_split(data, test_size=0.2)

    val_size = len(y_test)
    if len(y_train_full) > 2 * val_size:
        y_train, y_val = temporal_train_test_split(y_train_full, test_size=val_size)
    else:
        y_train, y_val = temporal_train_test_split(y_train_full, test_size=0.2)

    print(f"Dataset: {config['name']} | Cols: {data.shape[1]} | Full: {len(data)} | Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    return data, y_train_full, y_train, y_val, y_test