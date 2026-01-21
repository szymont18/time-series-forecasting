import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_fh
from .CATS import Model


class CatsConfig:
    """Mapuje parametry z forecastera na obiekt 'args' oczekiwany przez Model CATS."""

    def __init__(self, **kwargs):
        self.dec_in = kwargs.get("num_features", 1)
        self.seq_len = kwargs["seq_len"]
        # Używamy wewnętrznej, "naprawionej" długości predykcji (dla paddingu)
        self.pred_len = kwargs.get("internal_pred_len", kwargs["pred_len"])

        self.d_layers = kwargs.get("n_layers", 3)
        self.n_heads = kwargs.get("n_heads", 16)
        self.d_model = kwargs.get("d_model", 128)
        self.d_ff = kwargs.get("d_ff", 256)
        self.dropout = kwargs.get("dropout", 0.0)
        self.query_independence = kwargs.get("independence", False)
        self.patch_len = kwargs.get("patch_len", 24)
        self.stride = kwargs.get("stride", 24)
        self.padding_patch = kwargs.get("padding_patch", None)
        self.store_attn = kwargs.get("store_attn", False)
        self.QAM_start = kwargs.get("QAM_start", 0.1)
        self.QAM_end = kwargs.get("QAM_end", 0.5)


class LTSFCatsForecaster(BaseForecaster):
    _tags = {
        "scitype:y": "univariate",
        "scitype:X": "univariate",
        "y_inner_mtype": "pd.Series",
        "ignores-exogeneous-data": True,
    }

    def __init__(self, seq_len=96, pred_len=24, num_features=1, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, dropout=0.0, patch_len=24, stride=24, batch_size=32, num_epochs=20, lr=0.001,
                 verbose=True, **kwargs):  # Verbose domyślnie True

        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_len = patch_len
        self.stride = stride

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_patches = (self.pred_len + self.patch_len - 1) // self.patch_len
        self.internal_pred_len = num_patches * self.patch_len

        config_dict = self.__dict__.copy()
        config_dict['internal_pred_len'] = self.internal_pred_len

        args = CatsConfig(**config_dict)
        self.model = Model(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_sequences(self, y: pd.Series):
        data = y.values.astype(np.float32)
        X, Y = [], []

        if data.ndim == 1:
            data = data[:, np.newaxis]

        total_len = len(data)
        for i in range(total_len - self.seq_len - self.pred_len + 1):
            X_seq = data[i: i + self.seq_len]
            Y_seq = data[i + self.seq_len: i + self.seq_len + self.pred_len]
            X.append(X_seq)
            Y.append(Y_seq)

        return np.array(X), np.array(Y)

    def _fit(self, y, X=None, fh=None):
        X_train, Y_train = self._create_sequences(y)

        if len(X_train) == 0:
            self._is_fitted = True
            return self

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        epoch_iterator = range(1, self.num_epochs + 1)
        if self.verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training", unit="epoch")

        for epoch in epoch_iterator:
            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x)

                output = output[:, :self.pred_len, :]

                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            if self.verbose:
                avg_loss = epoch_loss / len(dataset)
                epoch_iterator.set_postfix(loss=f"{avg_loss:.6f}")

        self._is_fitted = True
        return self

    def _predict(self, fh, X=None):
        fh = check_fh(fh)
        y_pred_list = []
        y_window = self._y.iloc[-self.seq_len:].copy()
        max_fh = max(fh)
        step = 0

        while step < max_fh:
            input_data = y_window.values.astype(np.float32)[np.newaxis, :, np.newaxis]
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)

            self.model.eval()
            with torch.no_grad():
                pred_step = self.model(input_tensor)

            pred_step = pred_step[:, :self.pred_len, :]

            pred_step = pred_step.cpu().numpy().squeeze()
            if np.isscalar(pred_step):
                pred_step = np.array([pred_step])
            elif pred_step.ndim == 0:
                pred_step = pred_step.reshape(1)
            elif pred_step.ndim > 1:
                pred_step = pred_step.flatten()

            remaining = max_fh - step
            current_pred = pred_step[:remaining]
            y_pred_list.extend(current_pred)

            y_window = pd.concat([y_window, pd.Series(current_pred)], ignore_index=True)
            y_window = y_window[-self.seq_len:]
            step += len(current_pred)

        last_idx = self._y.index[-1]
        freq = None
        if hasattr(last_idx, "freq") and last_idx.freq is not None:
            freq = last_idx.freq
        elif hasattr(self._y.index, "freq") and self._y.index.freq is not None:
            freq = self._y.index.freq
        if freq is None and len(self._y.index) > 1:
            freq = pd.infer_freq(self._y.index)

        if freq is not None:
            fh_steps = np.array(fh).flatten()
            new_idx = [last_idx + (int(s) * freq) for s in fh_steps]
            y_index = pd.Index(new_idx)
        else:
            y_index = self._y.index[-1] + fh

        return pd.Series(y_pred_list, index=y_index)