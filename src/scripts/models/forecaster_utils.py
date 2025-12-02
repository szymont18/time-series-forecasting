import importlib
from enum import Enum
from typing import Type, Dict, Any
from sktime.forecasting.base import BaseForecaster

class ModelName(Enum):
    DLINEAR = "DLinear"
    CATS = "CATS"


MODEL_IMPORT_MAP = {
    ModelName.DLINEAR: ("sktime.forecasting.ltsf", "LTSFDLinearForecaster"),
    ModelName.CATS: ("src.scripts.models.ltsf_cats_forecaster", "LTSFCatsForecaster"),
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


def create_model_instance(model_name: ModelName, config_params: Dict[str, Any]) -> BaseForecaster:
    ForecasterClass = get_forecaster_class(model_name)

    forecaster_instance = ForecasterClass(**config_params)

    return forecaster_instance