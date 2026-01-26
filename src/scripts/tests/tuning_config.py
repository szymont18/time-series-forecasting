import optuna
from src.scripts.models.forecaster_utils import ModelName, get_forecaster_class

def get_common_seq_len(horizon):
    return min(horizon, 192)

def get_safe_seq_len_heavy(horizon):
    return min(horizon, 96)

def get_cats_params(trial: optuna.Trial, horizon, config=None):
    seq_len = get_common_seq_len(horizon)

    params = {
        "pred_len": horizon,
        "seq_len": seq_len,
        "stride": 24,
        "padding_patch": "end",
        "num_epochs": 6,
        "batch_size": 8,
        "patch_len": 24
    }

    if config:
        params.update(config)
    else:
        params["seq_len"] = trial.suggest_categorical("seq_len", [48, 96, 192, 336, 512])
        params["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        params["d_model"] = trial.suggest_categorical("d_model", [64, 128])
        params["n_heads"] = 4
        params["d_ff"] = trial.suggest_categorical("d_ff", [128, 256])
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)
        params["lr"] = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

    return params

def get_dlinear_params(trial: optuna.Trial, horizon, config=None):
    seq_len = get_common_seq_len(horizon)

    params = {
        "pred_len": horizon,
        "seq_len": seq_len,
        "num_epochs": 8,
        "batch_size": 32
    }

    if config:
        params.update(config)
    else:
        params["seq_len"] = trial.suggest_categorical("seq_len", [48, 96, 192, 336, 512])
        params["individual"] = trial.suggest_categorical("individual", [True, False])
        params["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    return params

def get_patchtst_params(trial: optuna.Trial, horizon, config=None):
    context_length = get_common_seq_len(horizon)

    base_config = {
        "context_length": context_length,
        "patch_length": 24,
        "patch_stride": 24,
        "prediction_length": horizon,
        "num_attention_heads": 4
    }

    base_training_args = {
        "num_train_epochs": 4,
        "per_device_train_batch_size": 8,
        "output_dir": "test_output_optuna",
        "overwrite_output_dir": True,
        "save_strategy": "no",
        "logging_strategy": "no",
    }

    if config:
        if "context_length" in config: base_config["context_length"] = config["context_length"]
        if "num_hidden_layers" in config: base_config["num_hidden_layers"] = config["num_hidden_layers"]
        if "d_model" in config: base_config["d_model"] = config["d_model"]
        if "head_dropout" in config: base_config["head_dropout"] = config["head_dropout"]
        if "learning_rate" in config: base_training_args["learning_rate"] = config["learning_rate"]
    else:
        base_config["context_length"] = trial.suggest_categorical("context_length", [48, 96, 192, 336, 512])
        base_config["num_hidden_layers"] = trial.suggest_int("num_hidden_layers", 2, 3)
        base_config["d_model"] = trial.suggest_categorical("d_model", [64, 128])
        base_config["ffn_dim"] = base_config["d_model"] * 2
        base_config["head_dropout"] = trial.suggest_float("head_dropout", 0.0, 0.2)
        base_training_args["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

    return {
        "config": base_config,
        "training_args": base_training_args
    }

def get_itransformer_params(trial: optuna.Trial, horizon, config=None):
    input_size = get_common_seq_len(horizon)

    params = {
        "input_size": input_size,
        "n_heads": 2,
        "factor": 1,
        "max_steps": 15,
        "scaler_type": 'standard',
        "n_series": 1,
        "batch_size": 8
    }

    if config:
        params.update(config)
    else:
        params["input_size"] = trial.suggest_categorical("input_size", [48, 96, 192, 336])
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128])
        params["d_ff"] = params["hidden_size"] * 2
        params["d_layers"] = trial.suggest_int("d_layers", 1, 2)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

    return params

def get_autoformer_params(trial: optuna.Trial, horizon, config=None):
    input_size = get_safe_seq_len_heavy(horizon)

    params = {
        "input_size": input_size,
        "n_head": 2,
        "batch_size": 2,
        "max_steps": 15
    }

    if config: params.update(config)
    else:
        params["input_size"] = trial.suggest_categorical("input_size", [48, 96])
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [16, 32])
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

    return params

def get_tft_params(trial: optuna.Trial, horizon, config=None):
    input_size = get_safe_seq_len_heavy(horizon)

    params = {
        "input_size": input_size,
        "max_steps": 15,
        "batch_size": 2,
    }

    if config:
        params.update(config)
    else:
        params["input_size"] = trial.suggest_categorical("input_size", [48, 96])
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [16, 32])
        params["dropout"] = trial.suggest_float("dropout", 0.1, 0.3)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

    return params

def get_chronos_params(trial: optuna.Trial, horizon, config=None):
    params = {
        "model_path": "amazon/chronos-t5-small",
        "config": {
            "num_samples": 20,
        }
    }

    if config:
        params["config"].update(config)
    else:
        params["config"]["temperature"] = trial.suggest_float("temperature", 0.1, 0.8)
        params["config"]["top_k"] = trial.suggest_int("top_k", 20, 80, step=20)
        params["config"]["top_p"] = trial.suggest_float("top_p", 0.8, 0.95)

    return params

def get_autoarima_params(trial: optuna.Trial, horizon, config=None):
    if config: return config
    return {"seasonal": trial.suggest_categorical("seasonal", [True, False])}

MODEL_SUGGEST_MAP = {
    ModelName.CATS: get_cats_params,
    ModelName.DLINEAR: get_dlinear_params,
    ModelName.PATCHTST: get_patchtst_params,
    ModelName.ITRANSFORMER: get_itransformer_params,
    ModelName.AUTOFORMER: get_autoformer_params,
    ModelName.TFT: get_tft_params,
    ModelName.CHRONOS: get_chronos_params,
    ModelName.AUTOARIMA: get_autoarima_params
}

MODELS_CONFIG = {
    ModelName.CATS.value: {
        "enum": ModelName.CATS, "class": get_forecaster_class(ModelName.CATS),
        "type": "sktime", "strategy": "optuna", "n_trials": 12,
        "refit_params": {"num_epochs": 30}
    },
    ModelName.DLINEAR.value: {
        "enum": ModelName.DLINEAR, "class": get_forecaster_class(ModelName.DLINEAR),
        "type": "sktime", "strategy": "optuna", "n_trials": 16,
        "refit_params": {"num_epochs": 40}
    },
    ModelName.PATCHTST.value: {
        "enum": ModelName.PATCHTST, "class": get_forecaster_class(ModelName.PATCHTST),
        "type": "sktime", "strategy": "optuna", "n_trials": 10,
        "refit_params": {"training_args": {"num_train_epochs": 25}}
    },
    ModelName.ITRANSFORMER.value: {
        "enum": ModelName.ITRANSFORMER, "class": get_forecaster_class(ModelName.ITRANSFORMER),
        "type": "neuralforecast", "strategy": "optuna", "n_trials": 10,
        "refit_params": {"max_steps": 40}
    },
    ModelName.AUTOFORMER.value: {
        "enum": ModelName.AUTOFORMER, "class": get_forecaster_class(ModelName.AUTOFORMER),
        "type": "neuralforecast", "strategy": "optuna", "n_trials": 6,
        "refit_params": {"max_steps": 40}
    },
    ModelName.TFT.value: {
        "enum": ModelName.TFT, "class": get_forecaster_class(ModelName.TFT),
        "type": "neuralforecast", "strategy": "optuna", "n_trials": 6,
        "refit_params": {"max_steps": 40}
    },
    ModelName.AUTOARIMA.value: {
        "enum": ModelName.AUTOARIMA, "class": get_forecaster_class(ModelName.AUTOARIMA),
        "type": "sktime", "strategy": "optuna", "n_trials": 3,
        "refit_params": {}
    },
    ModelName.CHRONOS.value: {
        "enum": ModelName.CHRONOS, "class": get_forecaster_class(ModelName.CHRONOS),
        "type": "sktime", "strategy": "optuna", "n_trials": 3,
        "refit_params": {"config": {"num_samples": 20}}
    },
}