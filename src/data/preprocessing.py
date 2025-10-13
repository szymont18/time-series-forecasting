# src/data/preprocess.py

import os
import glob
import modin.pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore


class Preprocessor:
    def __init__(self, source_dir="TSB/raw/TSB-U", target_dir="TSB/processed/TSB-U"):
        self.source_dir = source_dir
        self.target_dir = target_dir or source_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def _normalize(self, df):
        """
        Z-score normalization: (x - mean) / std
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols].apply(zscore)
        return df

    def process_all(self):
        """
        Process all CSVs in source_dir in parallel using Modin (multi-core).
        """
        files = glob.glob(os.path.join(self.source_dir, "*.csv"))
        if not files:
            print(f"No CSV files found in {self.source_dir}.")
            return

        print(f"Processing {len(files)} files from {self.source_dir}...")
        for f in tqdm(files):
            try:
                self._process_single(f)
            except Exception as e:
                print(f"⚠️ Error processing {f}: {e}")
        print("✅ All files processed.")



class UniVariatePreprocessor(Preprocessor):
    """
    Preprocess univariate time series datasets:
      1. Replace anomalies with previous valid values.
      2. Normalize data using z-score normalization.
      3. Save processed files back to target directory.
    """

    def __init__(self, source_dir="TSB/raw/TSB-U", target_dir="TSB/processed/TSB-U"):
        super().__init__(source_dir=source_dir, target_dir=target_dir)

    def _replace_anomalies(self, df):
        """
        Replace anomalies (label == 1) with previous non-anomalous value.
        """
        df.loc[df["Label"] == 1, "Data"] = None

        # Interpolate missing values (linear or time-based)
        df["Data"] = df["Data"].interpolate(method="linear")
        return df

    def _process_single(self, path):
        """
        Process one file end-to-end.
        """
        df = pd.read_csv(path)

        if "Data" not in df.columns or "Label" not in df.columns:
            raise ValueError(f"File {path} missing required columns.")

        df = self._replace_anomalies(df)
        df = self._normalize(df)
        df.drop(columns=["Label"], inplace=True)

        out_path = os.path.join(self.target_dir, os.path.basename(path))
        df.to_csv(out_path, index=False)
        return out_path



class MultiVariatePreprocessor(Preprocessor):
    """
    Preprocess multivariate time series datasets:
      1. Replace anomalies with interpolated values (linear) per column.
      2. Normalize each column using z-score.
      3. Save processed files back to target directory.
    """

    def __init__(self, source_dir="TSB/raw/TSB-MV", target_dir="TSB/processed/TSB-MV"):
        super().__init__(source_dir=source_dir, target_dir=target_dir)

    def _replace_anomalies(self, df):
        """
        Replace anomalies (Label==1) with linear interpolation for each data column.
        """
        # Identify label column
        if "Label" not in df.columns:
            raise ValueError("DataFrame must contain 'Label' column for anomalies")

        # Set anomalies to NaN in all numeric columns except Label
        data_columns = [c for c in df.columns if c != "Label"]
        df[data_columns] = df[data_columns].mask(df["Label"] == 1, np.nan)

        # Interpolate each column independently
        df[data_columns] = df[data_columns].interpolate(method="linear", axis=0)
        return df

    def _process_single(self, path):
        """
        Process one multivariate CSV file end-to-end.
        """
        df = pd.read_csv(path)

        if "Label" not in df.columns:
            raise ValueError(f"File {path} missing 'Label' column.")

        df = self._replace_anomalies(df)
        df = self._normalize(df)

        # Drop Label column after processing
        df.drop(columns=["Label"], inplace=True)

        out_path = os.path.join(self.target_dir, os.path.basename(path))
        df.to_csv(out_path, index=False)
        return out_path