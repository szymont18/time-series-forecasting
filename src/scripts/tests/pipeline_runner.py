import inspect
import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
import gc
import traceback
from typing import Optional

from neuralforecast import NeuralForecast
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError, MeanSquaredError, MeanAbsoluteScaledError, MeanAbsolutePercentageError
)
from sktime.utils.plotting import plot_series

from .tuning_config import MODELS_CONFIG, MODEL_SUGGEST_MAP
from .dataset_config import DATASETS
from .data_loader import load_and_split

OUT_DIR = "src/scripts/tests/out2"
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def wide_to_long_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.Series): df = df.to_frame()
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="h")

    df = df.reset_index()
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "ds"})

    df = pd.melt(df, id_vars=["ds"], var_name="unique_id", value_name="y")
    return df

def long_to_wide_df(df: pd.DataFrame, values_col: Optional[str] = None) -> pd.DataFrame:
    if "unique_id" not in df.columns:
        df = df.reset_index()
        df["unique_id"] = "unique_id"
    values_col = values_col if values_col else df.columns[-1]
    return pd.pivot(df, columns="unique_id", index="ds", values=values_col)

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def adjust_params_to_data_size(params, train_length, horizon, m_name):
    history_keys = ['seq_len', 'input_size', 'context_length']
    buffer = 2

    max_allowed_history = train_length - horizon - buffer
    if max_allowed_history < 24: max_allowed_history = 24

    for key in history_keys:
        val = None
        target_dict = params

        if key in params:
            val = params[key]
        elif "config" in params and key in params["config"]:
            val = params["config"][key]
            target_dict = params["config"]

        if val is not None:
            if val > max_allowed_history:
                target_dict[key] = int(max_allowed_history)

    return params

def calculate_metrics_dict(y_true, y_pred, y_train):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true.iloc[:min_len]
    y_pred = y_pred.iloc[:min_len]

    if isinstance(y_true, pd.Series): y_true = y_true.to_frame()
    if isinstance(y_pred, pd.Series): y_pred = y_pred.to_frame()
    if isinstance(y_train, pd.Series): y_train = y_train.to_frame()

    y_pred.index = y_true.index
    y_pred.columns = y_true.columns

    mae_raw = MeanAbsoluteError(multioutput='raw_values')(y_true, y_pred)
    mse_raw = MeanSquaredError(multioutput='raw_values')(y_true, y_pred)
    mape_raw = MeanAbsolutePercentageError(multioutput='raw_values')(y_true, y_pred)
    mase_raw = MeanAbsoluteScaledError(multioutput='raw_values')(y_true, y_pred, y_train=y_train)

    y_true_sum, y_pred_sum, y_train_sum = y_true.sum(axis=1), y_pred.sum(axis=1), y_train.sum(axis=1)
    mae_tot = MeanAbsoluteError()(y_true_sum, y_pred_sum)
    mse_tot = MeanSquaredError()(y_true_sum, y_pred_sum)
    mape_tot = MeanAbsolutePercentageError()(y_true_sum, y_pred_sum)
    mase_tot = MeanAbsoluteScaledError()(y_true_sum, y_pred_sum, y_train=y_train_sum)

    avg_dict = {
        "MAE": np.mean(mae_raw), "MSE": np.mean(mse_raw),
        "MAPE": np.mean(mape_raw), "MASE": np.mean(mase_raw)
    }

    return {
        "columns": y_true.columns,
        "mae_raw": mae_raw, "mse_raw": mse_raw, "mape_raw": mape_raw, "mase_raw": mase_raw,
        "total": {"MAE": mae_tot, "MSE": mse_tot, "MAPE": mape_tot, "MASE": mase_tot},
        "average": avg_dict
    }


def generate_result_string(dataset_label, model_name, horizon, metrics_data, best_params=None, dataset_size=None):
    size_info = f" (N={dataset_size})" if dataset_size else ""
    lines = [
        f"\n{'>' * 85}",
        f"DATA: {dataset_label}{size_info} | MODEL: {model_name} | HORIZON: {horizon}"
    ]
    if best_params:
        lines.append(f"BEST PARAMS: {best_params}")

    lines.append("-" * 85)
    lines.append(f"| {'Series':<12} | {'MAE':>10} | {'MSE':>10} | {'MAPE':>8} | {'MASE':>8}")
    lines.append("-" * 85)
    for i, col in enumerate(metrics_data["columns"]):
        lines.append(
            f"| {str(col)[:12]:<12} | {metrics_data['mae_raw'][i]:>10.4f} | {metrics_data['mse_raw'][i]:>10.4f} | {metrics_data['mape_raw'][i] * 100:>7.1f}% | {metrics_data['mase_raw'][i]:>8.4f}")
    lines.append("-" * 85)
    avg = metrics_data["average"]
    lines.append(
        f"| {'AVERAGE':<12} | {avg['MAE']:>10.4f} | {avg['MSE']:>10.4f} | {avg['MAPE'] * 100:>7.1f}% | {avg['MASE']:>8.4f}")
    t = metrics_data["total"]
    lines.append(
        f"| {'TOTAL SUM':<12} | {t['MAE']:>10.4f} | {t['MSE']:>10.4f} | {t['MAPE'] * 100:>7.1f}% | {t['MASE']:>8.4f}")
    lines.append(f"{'>' * 85}\n")
    return "\n".join(lines)

def reset_to_numeric_index(series):
    s = series.copy()
    s = s.reset_index(drop=True)
    return s

def prepare_plot_vectors(y_hist, y_true, y_pred):
    y_h = reset_to_numeric_index(y_hist)
    y_t = reset_to_numeric_index(y_true)
    y_p = reset_to_numeric_index(y_pred)
    start_future = len(y_h)
    y_t.index = range(start_future, start_future + len(y_t))
    y_p.index = range(start_future, start_future + len(y_p))
    return y_h, y_t, y_p

def calculate_step_for_limit(len_hist, len_future, limit=600):
    total_points = len_hist + len_future
    if total_points <= limit: return 1
    return int(np.ceil(total_points / limit))

def optuna_objective(trial, m_cfg, y_train, y_val, fh_val, ds_name, m_name):
    torch.cuda.empty_cache()
    gc.collect()

    val_horizon = int(len(y_val) * 0.20)
    if val_horizon < 12: val_horizon = 12

    params = MODEL_SUGGEST_MAP[m_cfg["enum"]](trial, horizon=val_horizon)
    params = adjust_params_to_data_size(params, len(y_train), val_horizon, m_name)

    try:
        cls = m_cfg["class"]

        if m_cfg["type"] == "sktime":
            sig = inspect.signature(cls.__init__)
            if "verbose" in sig.parameters: params["verbose"] = False

            if "seq_len" in params and params.get("seq_len", 0) >= len(y_train):
                return float('inf')

            model = cls(**params)
            model.fit(y=y_train, fh=list(range(1, val_horizon + 1)))
            y_pred = model.predict(fh=list(range(1, val_horizon + 1)))
        else:
            params["h"] = val_horizon
            params["n_series"] = y_train.shape[1] if len(y_train.shape) > 1 else 1

            nf = NeuralForecast(models=[cls(**params)], freq='D')
            nf.fit(df=wide_to_long_df(y_train), verbose=False)
            y_pred = long_to_wide_df(nf.predict()).iloc[:val_horizon]

        y_val_sliced = y_val.iloc[:val_horizon]

        loss = MeanSquaredError()(y_val_sliced, y_pred)
        if np.isnan(loss) or np.isinf(loss): return float('inf')
        return loss

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[TRIAL ERROR] CUDA OOM. Cleaning cache.")
            torch.cuda.empty_cache()
            gc.collect()
            return float('inf')
        return float('inf')
    except Exception as e:
        return float('inf')

def run_tuning_pipeline():
    for ds_conf in DATASETS:
        if ds_conf.get("skip"):
            print(f"Skipping dataset: {ds_conf['name']}")
            continue

        torch.cuda.empty_cache()
        gc.collect()

        try:
            data, y_train_full, y_train, y_val, y_test = load_and_split(ds_conf)
        except Exception as e:
            print(f"Failed to load dataset {ds_conf['name']}: {e}")
            continue

        real_horizon = len(y_test)

        ds_out_dir = os.path.join(OUT_DIR, ds_conf["name"])
        summary_path = os.path.join(ds_out_dir, "all_results.txt")

        for m_name, cfg in MODELS_CONFIG.items():
            print(f"\n>>> Running {m_name} on {ds_conf['name']}")
            m_dir = os.path.join(ds_out_dir, m_name)
            os.makedirs(m_dir, exist_ok=True)

            try:
                best_params_raw = {}
                if cfg["strategy"] == "optuna":
                    study = optuna.create_study(direction="minimize")
                    study.optimize(lambda t: optuna_objective(t, cfg, y_train, y_val, None, ds_conf["name"], m_name),
                                   n_trials=cfg["n_trials"])
                    best_params_raw = study.best_params

                current_horizon = real_horizon
                best_params = MODEL_SUGGEST_MAP[cfg["enum"]](None, horizon=current_horizon, config=best_params_raw)
                best_params = deep_update(best_params, cfg.get("refit_params", {}))
                best_params = adjust_params_to_data_size(best_params, len(y_train_full), current_horizon, m_name)

                cls = cfg["class"]

                if cfg["type"] == "sktime":
                    sig = inspect.signature(cls.__init__)
                    if "verbose" in sig.parameters: best_params["verbose"] = True

                    final_model = cls(**best_params)
                    final_model.fit(y=y_train_full, fh=list(range(1, current_horizon + 1)))
                    y_pred = final_model.predict(fh=list(range(1, current_horizon + 1)))
                else:
                    best_params["h"] = current_horizon
                    best_params["n_series"] = y_train_full.shape[1] if len(y_train_full.shape) > 1 else 1

                    final_model = NeuralForecast(models=[cls(**best_params)], freq='D')
                    final_model.fit(df=wide_to_long_df(y_train_full), verbose=True)
                    y_pred = long_to_wide_df(final_model.predict()).iloc[:current_horizon]

                y_test_sliced = y_test.iloc[:current_horizon]
                y_pred.index = y_test_sliced.index

                metrics = calculate_metrics_dict(y_test_sliced, y_pred, y_train_full)
                res_str = generate_result_string(ds_conf["name"], m_name, current_horizon, metrics, best_params, dataset_size=len(data))

                with open(os.path.join(m_dir, "best_params.json"), "w") as f:
                    json.dump(best_params, f, indent=4, default=str)
                with open(os.path.join(m_dir, "metrics.txt"), "w") as f:
                    f.write(res_str)
                with open(summary_path, "a") as f:
                    f.write(res_str)

                plot_dir = os.path.join(m_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

                y_test_plot = y_test_sliced.to_frame() if isinstance(y_test_sliced, pd.Series) else y_test_sliced

                for col in y_test_plot.columns:
                    y_hist_raw = y_train_full[col]
                    y_true_raw = y_test_sliced[col]
                    y_pred_raw = y_pred[col]

                    y_h_clean, y_t_clean, y_p_clean = prepare_plot_vectors(y_hist_raw, y_true_raw, y_pred_raw)
                    step = calculate_step_for_limit(len(y_h_clean), len(y_t_clean), limit=600)

                    plt.figure(figsize=(10, 4))
                    plot_series(y_h_clean.iloc[::step], y_t_clean.iloc[::step], y_p_clean.iloc[::step],
                                labels=["History", "True", "Pred"])
                    safe_col_name = str(col).replace("/", "_").replace("\\", "_")
                    plt.title(f"Full Forecast | {ds_conf['name']} | {m_name}")
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(os.path.join(plot_dir, f"{safe_col_name}_full.png"))
                    plt.close()

                    plt.figure(figsize=(10, 4))
                    plot_series(y_h_clean.iloc[-100:], y_t_clean.iloc[:200], y_p_clean.iloc[:200],
                                labels=["History", "True", "Pred"])
                    plt.title(f"Zoom | {ds_conf['name']} | {m_name}")
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(os.path.join(plot_dir, f"{safe_col_name}_zoom.png"))
                    plt.close()

                y_train_sum = y_train_full.sum(axis=1) if isinstance(y_train_full, pd.DataFrame) else y_train_full
                y_test_sum = y_test_sliced.sum(axis=1) if isinstance(y_test_sliced, pd.DataFrame) else y_test_sliced
                y_pred_sum = y_pred.sum(axis=1) if isinstance(y_pred, pd.DataFrame) else y_pred
                y_h_s, y_t_s, y_p_s = prepare_plot_vectors(y_train_sum, y_test_sum, y_pred_sum)
                step_sum = calculate_step_for_limit(len(y_h_s), len(y_t_s), limit=600)

                plt.figure(figsize=(10, 4))
                plot_series(y_h_s.iloc[::step_sum], y_t_s.iloc[::step_sum], y_p_s.iloc[::step_sum],
                            labels=["Hist", "True", "Pred"])
                plt.title(f"AGGREGATED TOTAL | {ds_conf['name']} | {m_name}")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(plot_dir, "TOTAL_SUM_downsampled.png"))
                plt.close()

                print(res_str)

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Pipeline failed for {m_name} on {ds_conf['name']}: {e}")
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    run_tuning_pipeline()