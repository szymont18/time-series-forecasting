from matplotlib import pyplot as plt
import os
import pickle

from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.utils import plot_series
from sktime.split import temporal_train_test_split

from src.scripts.forecasting_utils import load_data_from_file
from src.scripts.models.forecaster_utils import ModelName, DEFAULT_PARAMS, create_model_instance

TRAINED_MODELS_DIR = "trained_models/"
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


def make_dlinear(seq_len: int, pred_len: int):
    params = DEFAULT_PARAMS[ModelName.DLINEAR]
    params["seq_len"] = seq_len
    params["pred_len"] = pred_len
    return create_model_instance(ModelName.DLINEAR, params)


def make_cats(seq_len: int, pred_len: int):
    params = DEFAULT_PARAMS[ModelName.CATS]
    params["seq_len"] = seq_len
    params["pred_len"] = pred_len
    return create_model_instance(ModelName.CATS, params)


def make_patchtst(seq_len: int, pred_len: int):
    params = DEFAULT_PARAMS[ModelName.PATCHTST]
    params["config"]["context_length"] = seq_len
    params["config"]["prediction_length"] = pred_len
    return create_model_instance(ModelName.PATCHTST, params)


def make_chronos(seq_len: int, pred_len: int):
    return create_model_instance(ModelName.CHRONOS)


def train_models(file_path: str, seq_pred_combinations=None, max_points: int = 1000):
    raw_data = load_data_from_file(file_path)[:max_points]
    train_data, test_data = temporal_train_test_split(raw_data, test_size=0.2)
    fh = list(range(1, len(test_data) + 1))

    # Default combinations
    if seq_pred_combinations is None:
        seq_pred_combinations = [
			(24, 24),  # 1 patch
			(48, 24),  # 2 patches
			(96, 24),  # 4 patches
		]

    trained_models = []

    try:
        chronos_model = make_chronos(0, 0)
        chronos_model.fit(y=train_data, fh=fh)
        model_path = os.path.join(TRAINED_MODELS_DIR, "CHRONOS.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(chronos_model, f)
        trained_models.append(model_path)
        print("CHRONOS trained and saved.")
    except Exception as e:
        print(f"Error training CHRONOS: {e}")

    model_funcs = {ModelName.DLINEAR: make_dlinear, ModelName.CATS: make_cats, ModelName.PATCHTST: make_patchtst}
    for model_name, make_func in model_funcs.items():
        for seq_len, pred_len in seq_pred_combinations:
            try:
                model = make_func(seq_len=seq_len, pred_len=pred_len)
                model.fit(y=train_data, fh=fh)
                model_filename = f"{model_name}_{seq_len}_{pred_len}.pkl"
                model_path = os.path.join(TRAINED_MODELS_DIR, model_filename)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                trained_models.append(model_path)
                print(f"{model_name} (seq_len={seq_len}, pred_len={pred_len}) trained and saved.")
            except Exception as e:
                print(f"Error training {model_name} with seq_len={seq_len}, pred_len={pred_len}: {e}")

    return trained_models



def evaluate_models(file_path: str, model_paths=None, max_points: int = 1000, out_dir: str = "plots"):
    raw_data = load_data_from_file(file_path)[:max_points]
    train_data, test_data = temporal_train_test_split(raw_data, test_size=0.2)
    fh = list(range(1, len(test_data) + 1))

    os.makedirs(out_dir, exist_ok=True)

    if model_paths is None:
        model_paths = [os.path.join(TRAINED_MODELS_DIR, f) for f in os.listdir(TRAINED_MODELS_DIR) if f.endswith(".pkl")]

    results = []

    for model_path in model_paths:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(fh=fh)

            mse = MeanSquaredError()(test_data, y_pred)
            mae = MeanAbsoluteError()(test_data, y_pred)

            model_name = os.path.splitext(os.path.basename(model_path))[0]
            results.append({"model": model_name, "mse": float(mse), "mae": float(mae)})

            # Plot
            plt.figure(figsize=(10, 4))
            plot_series(raw_data, y_pred, labels=["True", "Predicted"])
            plt.title(f"{model_name} Forecast")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model_name}_forecast.png"))
            plt.clf()

            print(f"Evaluated {model_name}. MSE={mse:.4f}, MAE={mae:.4f}")
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")

    # Summary
    if results:
        print("\nSummary metrics:")
        for r in results:
            print(f" - {r['model']}: MSE={r['mse']:.4f}, MAE={r['mae']:.4f}")
    else:
        print("No results to show.")



if __name__ == '__main__':
    file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
    
    train_models(file_path=file_path, max_points=1000)

    evaluate_models(file_path=file_path, max_points=1000, out_dir="plots")
