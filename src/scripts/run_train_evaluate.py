from matplotlib import pyplot as plt
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanAbsoluteScaledError
from sktime.utils import plot_series

from src.scripts.forecasting_utils import load_data_from_file
from src.scripts.models.forecaster_utils import create_model_instance, ModelName
from sktime.split import temporal_train_test_split


if __name__ == '__main__':
    file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
    raw_data = load_data_from_file(file_path)[:1000]

    train_data, test_data = temporal_train_test_split(raw_data, test_size=0.2)
    fh = list(range(1, len(test_data) + 1))

    cats_model = create_model_instance(model_name=ModelName.CHRONOS)

    cats_model.fit(y=train_data, fh=fh)

    y_pred = cats_model.predict(fh=fh)

    mae = MeanAbsoluteError()(test_data, y_pred)
    mase = MeanAbsoluteScaledError()(test_data, y_pred, y_train=train_data)
    print(f"MAE: {mae:.4f}")
    print(f"MASE: {mase:.4f}")

    plot_series(raw_data, y_pred, labels=["True", "Predicted"])
    plt.title("CATS Forecast Evaluation")
    plt.show()
    plt.clf()



