import importlib
from enum import Enum
from typing import Type, Dict, Any
from neuralforecast import NeuralForecast
from sktime.forecasting.base import BaseForecaster
import pandas as pd
from sktime.split import temporal_train_test_split

class ModelName(Enum):
    # Without Transformers
    DLINEAR = "DLinear"
    CATS = "CATS"
    CHRONOS = "CHRONOS"
    AUTOARIMA = "AUTOARIMA"

    # With Transformers
    PATCHTST = "PATCHTST"
    ITRANSFORMER = "ITRANSFORMER"
    AUTOFORMER = "AUTOFORMER"
    TFT = "TEMPORALFUSIONTRANSFORMER"


NEURAL_FORECAST_MODELS = [ModelName.TFT, ModelName.AUTOFORMER, ModelName.ITRANSFORMER]

MODEL_IMPORT_MAP = {
    ModelName.DLINEAR: ("sktime.forecasting.ltsf", "LTSFDLinearForecaster"),
    ModelName.CATS: ("src.scripts.models.ltsf_cats_forecaster", "LTSFCatsForecaster"),
    ModelName.PATCHTST: ("sktime.forecasting.patch_tst", "PatchTSTForecaster"),
    ModelName.CHRONOS: ("sktime.forecasting.chronos", "ChronosForecaster"),
    ModelName.ITRANSFORMER: ("neuralforecast.models", "iTransformer"),
    ModelName.AUTOFORMER: ("neuralforecast.models", "Autoformer"),
    ModelName.TFT: ("neuralforecast.models", "TFT"),
    ModelName.AUTOARIMA: ("sktime.forecasting.statsforecast", "StatsForecastAutoARIMA")
}

DEFAULT_PARAMS = {
    ModelName.DLINEAR: {"seq_len": 96,
                        "pred_len": 24,
                        "individual": False,
                        "batch_size": 32,
                        "num_epochs": 100,
                        "lr": 0.01},
    ModelName.CATS: {"seq_len": 96,
                     "pred_len": 24,
                     "batch_size": 32,
                     "num_epochs": 200,
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
                     "padding_patch": "end"},
    ModelName.PATCHTST: {
        "config": {
                "context_length": 96,
                "patch_stride": 24,
                "d_model": 128,
                "num_attention_heads": 8,
                "ffn_dim": 256,
                "head_dropout": 0.3,
                "num_hidden_layers": 1,
                "patch_length": 24,
                "prediction_length": 24
            },
        "training_args" : {
            "learning_rate": 1e-4,
            "num_train_epochs": 2000,
            "per_device_train_batch_size": 16,
        }
    },
    ModelName.CHRONOS: {
        "model_path": "amazon/chronos-t5-small",
        "config": {
            "num_samples": 128,
            "temperature": 0.3,
            "top_k": 100,
            "top_p": 0.2
        }
    },

    ModelName.ITRANSFORMER: {
        "h": 200,
        "n_series": 1,
        "input_size": 96,
        "hidden_size": 128,
        "n_heads": 2,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 4,
        "factor": 1,
    },

    ModelName.AUTOFORMER: {
        "h": 200,
        "input_size": 24,
        "hidden_size": 16,
        "conv_hidden_size": 32,
        "n_head": 2,
        "max_steps": 300
    },

    ModelName.TFT: {
        "h": 200,
        "input_size": 48,
        "hidden_size": 20,
        "grn_activation": "ELU",
        "rnn_type": "lstm",
        "n_rnn_layers": 1,
        "max_steps": 300
    },

    ModelName.AUTOARIMA: {
        "seasonal": True
    }
}

def wide_to_long_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).to_frame(name="y")

    df["ds"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="D"
    )

    df["unique_id"] = "series_1"

    return df[["unique_id", "ds", "y"]]
def get_train_test(model_name: ModelName, raw_data: pd.DataFrame):
    train_data, test_data = temporal_train_test_split(raw_data, test_size=0.2)

    if model_name in NEURAL_FORECAST_MODELS:
        train_data, test_data = wide_to_long_df(train_data), wide_to_long_df(test_data)

    return train_data, test_data

def fit_predict(model, model_name: ModelName, train_data: pd.DataFrame, test_data: pd.DataFrame):
    if model_name in NEURAL_FORECAST_MODELS:
        model.fit(df=train_data)
        y_pred = model.predict(test_data)
        y_pred = pd.Series(
            y_pred.iloc[:, -1].values,
            index=range(len(train_data), len(train_data) + len(y_pred))
        )
        return y_pred

    fh = list(range(1, len(test_data) + 1))
    model.fit(y=train_data, fh=fh)

    y_pred = model.predict(fh=fh)
    return y_pred

def get_forecaster_class(model_name: ModelName) -> Type[BaseForecaster]:
    if model_name not in MODEL_IMPORT_MAP:
        raise ValueError(f"Nieznany model: {model_name.value}")

    module_name, class_name = MODEL_IMPORT_MAP[model_name]

    try:
        module = importlib.import_module(module_name)
        forecaster_class = getattr(module, class_name)
        return forecaster_class

    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Błąd ładowania klasy {class_name} z modułu {module_name}. Sprawdź pliki: {e}")


def create_evaluation(model_name: ModelName, raw_data: pd.DataFrame, config_params: Dict[str, Any] = None):
    ForecasterClass = get_forecaster_class(model_name)

    if config_params is None:
        config_params = DEFAULT_PARAMS[model_name]

    forecaster_instance = ForecasterClass(**config_params)
    if model_name in NEURAL_FORECAST_MODELS:
        forecaster_instance = NeuralForecast(models=[forecaster_instance], freq='D')
    train_data, test_data = get_train_test(model_name, raw_data)

    return forecaster_instance, train_data, test_data

def create_model_instance(model_name: ModelName, config_params: Dict[str, Any] = None) -> BaseForecaster:
    ForecasterClass = get_forecaster_class(model_name)

    if config_params is None:
        config_params = DEFAULT_PARAMS[model_name]

    forecaster_instance = ForecasterClass(**config_params)

    return forecaster_instance