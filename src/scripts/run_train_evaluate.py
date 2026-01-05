import os

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from matplotlib import pyplot as plt
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanAbsoluteScaledError
from sktime.utils import plot_series

from src.scripts.forecasting_utils import load_data_from_file
from src.scripts.models.forecaster_utils import create_evaluation, ModelName, fit_predict


if __name__ == '__main__':
    file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
    raw_data = load_data_from_file(file_path)[:1000]
    model_type = ModelName.TFT

    model, train_data, test_data = create_evaluation(model_name=model_type, raw_data=raw_data)
    y_pred = fit_predict(model, model_type, train_data, test_data)

    if not isinstance(train_data, pd.Series):
        train_data = train_data["y"]
        test_data = test_data["y"]

    mae = MeanAbsoluteError()(test_data, y_pred)
    print(f"MAE: {mae:.4f}")
    mase = MeanAbsoluteScaledError()(test_data, y_pred, y_train=train_data)
    print(f"MASE: {mase:.4f}")

    plot_series(raw_data, y_pred, labels=["True", "Predicted"])
    plt.title("CATS Forecast Evaluation")
    plt.show()
    plt.clf()



