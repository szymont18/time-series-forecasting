import pandas as pd
import numpy as np

def preprocess_daily(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

    df = df.asfreq('D')

    df = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    return df

def preprocess_simple(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna()
    return df

DATASETS = [
    {
        "name": "Finance-U",
        "source": "file",
        "path": "resources/data/TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv",
        "preprocess": preprocess_daily,
        "type": "U",
        "skip": False
    },
    {
        "name": "WebService-U",
        "source": "file",
        "path": "resources/data/TSB/processed/TSB-U/079_WSD_id_51_WebService_tr_4559_1st_10672.csv",
        "preprocess": preprocess_daily,
        "type": "U",
        "skip": False
    },
    {
        "name": "Facility-U",
        "source": "file",
        "path": "resources/data/TSB/processed/TSB-U/817_Exathlon_id_8_Facility_tr_10766_1st_12590.csv",
        "preprocess": preprocess_daily,
        "type": "U",
        "skip": False
    },
    {
        "name": "Facility-M",
        "source": "file",
        "path": "resources/data/TSB/processed/TSB-M/193_Exathlon_id_20_Facility_tr_8898_1st_8998.csv",
        "preprocess": preprocess_daily,
        "type": "M",
        "skip": False
    },
    {
        "name": "Environment-M",
        "source": "file",
        "path": "resources/data/TSB/processed/TSB-M/124_TAO_id_9_Environment_tr_500_1st_1.csv",
        "preprocess": preprocess_daily,
        "type": "M",
        "skip": False
    },
    {
        "name": "Medical-M",
        "source": "file",
        "path": "resources/data/TSB/processed/TSB-M/023_MITDB_id_5_Medical_tr_25000_1st_36913.csv",
        "preprocess": preprocess_daily,
        "type": "M",
        "skip": False
    }
]