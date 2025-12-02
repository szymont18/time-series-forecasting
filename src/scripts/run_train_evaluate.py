from matplotlib import pyplot as plt
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanAbsoluteScaledError
from sktime.utils import plot_series

from .forecasting_utils import load_data_from_file
from .models.forecaster_utils import create_model_instance, ModelName

if __name__ == '__main__':
    file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
    raw_data = load_data_from_file(file_path)[:1000]

    train_size = int(0.8 * len(raw_data))
    train_data = raw_data[:train_size]
    test_data = raw_data[train_size:]

    fh = list(range(1, len(test_data) + 1))

    config = {
        "seq_len": 96,
        "pred_len": 24,
        "batch_size": 32,
        "num_epochs": 400,
        "lr": 0.0001,
        "num_features": 1,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 8,
        "patch_len": 24,
        "stride": 24,
        "dropout": 0.1,
        "d_ff": 256,
        "activation": "gelu",
        "padding_patch": "end"
    }

    cats_model = create_model_instance(model_name=ModelName.CATS, config_params=config)

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



#if __name__ == '__main__':
#    file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
#    raw_data = load_data_from_file(file_path)
#
#    PRED_LEN = 1
#
#    train_size = int(0.8 * len(raw_data))
#    train_data = raw_data[:train_size]
#    test_data = raw_data[train_size:]
#
#    fh = list(range(1, len(test_data) + 1))
#
#    config = {
#        "seq_len": 96,
#        "pred_len": PRED_LEN,
#        "individual": False,
#        "batch_size": 32,
#        "num_epochs": 2,
#        "lr": 0.01
#    }
#
#
#    dlinear_model = create_model_instance(model_name=ModelName.DLINEAR, config_params=config)
#
#    dlinear_model.fit(y=train_data, fh=fh)
#
#    y_pred = dlinear_model.predict(fh=fh)
#
#    mae = MeanAbsoluteError()(test_data, y_pred)
#    mase = MeanAbsoluteScaledError()(test_data, y_pred, y_train=train_data)
#    print(f"MAE: {mae:.4f}")
#    print(f"MASE: {mase:.4f}")
#
#    plot_series(raw_data, y_pred, labels=["True", "Predicted"])
#    plt.title("DLinear Forecast Evaluation")
#    plt.show()
#    plt.clf()


