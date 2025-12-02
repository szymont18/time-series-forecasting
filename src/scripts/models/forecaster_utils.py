import importlib
from enum import Enum
from typing import Type, Dict, Any
from sktime.forecasting.base import BaseForecaster

class ModelName(Enum):
    DLINEAR = "DLinear"
    CATS = "CATS"
    PATCHTST = "PATCHTST"


MODEL_IMPORT_MAP = {
    ModelName.DLINEAR: ("sktime.forecasting.ltsf", "LTSFDLinearForecaster"),
    ModelName.CATS: ("src.scripts.models.ltsf_cats_forecaster", "LTSFCatsForecaster"),
    ModelName.PATCHTST: ("sktime.forecasting.patch_tst", "PatchTSTForecaster")
}

DEFAULT_PARAMS = {
    ModelName.DLINEAR: {},
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
    }
}

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


def create_model_instance(model_name: ModelName, config_params: Dict[str, Any] = None) -> BaseForecaster:
    ForecasterClass = get_forecaster_class(model_name)

    if config_params is None:
        config_params = DEFAULT_PARAMS[model_name]

    forecaster_instance = ForecasterClass(**config_params)

    return forecaster_instance