import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.scripts.models import DLinear
import torch.nn as nn

matplotlib.use('TkAgg')

SEQ_LEN = 96
PRED_LEN = 24
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_data_from_file(file_path):
    """Wczytuje dane i zwraca serię czasową jako NumPy array (wartości)."""
    df = pd.read_csv(file_path, parse_dates=['Data'])
    data = df["Data"].values.reshape(-1, 1)
    return data


def calculate_metrics(actuals_final, predictions_final):
    """Oblicza i wyświetla podstawowe metryki ewaluacyjne (MAE, RMSE)."""
    mae = mean_absolute_error(actuals_final.flatten(), predictions_final.flatten())
    rmse = np.sqrt(mean_squared_error(actuals_final.flatten(), predictions_final.flatten()))
    print(f"\nWyniki na zbiorze testowym (wszystkie sekwencje):")
    print(f"MAE (Średni Błąd Absolutny): {mae:.2f}")
    print(f"RMSE (Pierwiastek Błędu Średnio-Kwadratowego): {rmse:.2f}")


def plot_data_series(actuals_final: np.ndarray, predictions_final: np.ndarray,
                     title: str = 'Porównanie Rzeczywistych Wartości i Prognoz DLinear (Cały Zbiór Testowy)'):
    """Rysuje porównanie rzeczywistych wartości i prognoz dla całego spłaszczonego zbioru testowego."""
    y_actual = actuals_final.flatten()
    y_pred = predictions_final.flatten()
    time_index = np.arange(len(y_actual))

    plt.figure(figsize=(18, 7))
    plt.plot(time_index, y_actual, label='Wartość Rzeczywista (Y)', color='#1f77b4', linewidth=2, alpha=0.8)
    plt.plot(time_index, y_pred, label='Prognoza (Y_Pred)', color='#ff7f0e', linestyle='--', linewidth=1.5)

    plt.title(title, fontsize=16)
    plt.xlabel('Indeks Czasowy (punkty danych w zbiorze testowym)', fontsize=12)
    plt.ylabel('Wartość', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_single_forecast_window(X_train_t: torch.Tensor, actuals_final: np.ndarray, predictions_final: np.ndarray,
                                seq_len: int, pred_len: int, scaler: MinMaxScaler):
    """
    Rysuje wykres skupiający się na końcówce danych:
    Ostatnia historia (X_train) + Pierwsza Prognoza (Y_pred[0]) vs. Pierwsza Rzeczywistość (Y_actual[0]).
    """
    last_x_train_scaled = X_train_t[-1].cpu().numpy().reshape(-1, 1)
    last_x_train = scaler.inverse_transform(last_x_train_scaled).flatten()

    pred_sequence = predictions_final[0].flatten()
    actual_sequence = actuals_final[0].flatten()

    full_actual = np.concatenate([last_x_train, actual_sequence])
    full_prediction = np.concatenate([last_x_train, pred_sequence])
    time_index = np.arange(len(full_actual))

    plt.figure(figsize=(15, 6))

    plt.plot(time_index, full_actual, label='Historia + Rzeczywista Przyszłość (Y)', color='#1f77b4', linewidth=2)
    plt.plot(time_index, full_prediction, label='Prognoza Modelu', color='#ff7f0e', linestyle='--', linewidth=1.5)

    plt.axvline(x=seq_len - 1, color='red', linestyle=':', label='Koniec Historii (Początek Prognozy)', linewidth=2)
    plt.text(seq_len, np.max(full_actual), ' Obszar Prognozy', fontsize=12, color='red')

    plt.title(f'Wizualizacja Ostatniej Prognozy (Historia: {seq_len}, Prognoza: {pred_len})', fontsize=16)
    plt.xlabel('Indeks Czasowy')
    plt.ylabel('Wartość (odwrócona normalizacja)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()



class UniversalForecaster:
    """
    Uniwersalna klasa do trenowania i ewaluacji modeli prognozowania szeregów czasowych PyTorch.
    Akceptuje dowolny model dziedziczący po nn.Module w konstruktorze.
    """

    def __init__(self, model: nn.Module, data: np.ndarray, seq_len: int, pred_len: int, batch_size: int,
                 train_ratio: float):

        self.model = model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.data = data

        self.scaler = MinMaxScaler()
        self.device = DEVICE
        self.model.to(self.device)

        self.X_train_t = None
        self.X_test = None
        self.Y_test = None
        self.train_loader = None

        self._prepare_data(train_ratio)

    def _create_sequences(self, data: np.ndarray):
        """Tworzy okna wejściowe (X) i wyjściowe (Y)."""
        X, Y = [], []
        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            x = data[i:i + self.seq_len]
            y = data[i + self.seq_len:i + self.seq_len + self.pred_len]
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def _prepare_data(self, train_ratio: float):
        """Dzieli, skaluje i tworzy DataLoader."""

        train_size = int(len(self.data) * train_ratio)
        train_data = self.data[:train_size]
        self.scaler.fit(train_data)
        data_scaled = self.scaler.transform(self.data)

        X_all, Y_all = self._create_sequences(data_scaled)

        num_sequences = len(X_all)
        train_end_index = int(num_sequences * train_ratio)

        X_train = X_all[:train_end_index]
        Y_train = Y_all[:train_end_index]
        self.X_test = X_all[train_end_index:]
        self.Y_test = Y_all[train_end_index:]

        self.X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
        train_dataset = TensorDataset(self.X_train_t, Y_train_t)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs: int = 10, learning_rate: float = 0.001):
        """Trenuje przekazany model."""

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print("Początek treningu")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")
        print("Trening zakończony!")

    def evaluate(self):
        """Generuje prognozy na zbiorze testowym i odwraca skalowanie."""

        self.model.eval()
        X_test_t = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions_scaled = self.model(X_test_t).cpu().numpy()

        predictions_reshaped = predictions_scaled.reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions_reshaped)
        predictions_final = predictions.reshape(predictions_scaled.shape)

        actuals_reshaped = self.Y_test.reshape(-1, 1)
        actuals_final = self.scaler.inverse_transform(actuals_reshaped).reshape(self.Y_test.shape)

        return actuals_final, predictions_final



if __name__ == '__main__':
    file_path = "TSB/processed/TSB-U/149_Stock_id_1_Finance_tr_500_1st_7.csv"
    raw_data = load_data_from_file(file_path)

    dlinear_model = DLinear(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        individual=False,
        num_features=1
    )

    forecaster = UniversalForecaster(
        model=dlinear_model,
        data=raw_data,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO
    )

    forecaster.train(num_epochs=10, learning_rate=0.001)

    actuals_final, predictions_final = forecaster.evaluate()

    calculate_metrics(actuals_final, predictions_final)

    plot_single_forecast_window(
        forecaster.X_train_t,
        actuals_final,
        predictions_final,
        SEQ_LEN,
        PRED_LEN,
        forecaster.scaler
    )

    #plot_data_series(actuals_final, predictions_final)