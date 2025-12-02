from matplotlib import pyplot as plt
import os

from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.utils import plot_series
from sktime.split import temporal_train_test_split

from src.scripts.forecasting_utils import load_data_from_file
from src.scripts.models.forecaster_utils import create_model_instance, ModelName


def evaluate_models(file_path: str, max_points: int = 1000, out_dir: str = "plots"):
	raw_data = load_data_from_file(file_path)[:max_points]

	train_data, test_data = temporal_train_test_split(raw_data, test_size=0.2)
	fh = list(range(1, len(test_data) + 1))

	os.makedirs(out_dir, exist_ok=True)

	results = []

	model_names = [ModelName.DLINEAR, ModelName.CATS, ModelName.PATCHTST, ModelName.CHRONOS]

	for m in model_names:
		print(f"\nEvaluating model: {m.value}")
		try:
			model = create_model_instance(model_name=m)
		except Exception as e:
			print(f"  Skipping {m.value} - failed to create instance: {e}")
			continue

		try:
			model.fit(y=train_data, fh=fh)
			y_pred = model.predict(fh=fh)

			mse = MeanSquaredError()(test_data, y_pred)
			mae = MeanAbsoluteError()(test_data, y_pred)

			print(f"  MSE: {mse:.4f}")
			print(f"  MAE: {mae:.4f}")

			results.append({"model": m.value, "mse": float(mse), "mae": float(mae)})

			# Plot and save
			plt.figure(figsize=(10, 4))
			plot_series(raw_data, y_pred, labels=["True", "Predicted"])
			plt.title(f"{m.value} Forecast")
			out_path = os.path.join(out_dir, f"{m.value}_forecast.png")
			plt.tight_layout()
			plt.savefig(out_path)
			plt.clf()
			print(f"  Saved plot to: {out_path}")

		except Exception as e:
			print(f"  Error evaluating {m.value}: {e}")
			continue

	# Summary
	if results:
		print("\nSummary metrics:")
		for r in results:
			print(f" - {r['model']}: MSE={r['mse']:.4f}, MAE={r['mae']:.4f}")
	else:
		print("No results to show.")


if __name__ == '__main__':
	file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
	evaluate_models(file_path=file_path, max_points=1000, out_dir="plots")

