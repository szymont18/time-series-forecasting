import torch
import torch.nn as nn


class Decomposition(nn.Module):
    def __init__(self, kernel_size):
        super(Decomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        trend = self.moving_avg(x_permuted).permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, individual, num_features, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = Decomposition(kernel_size)

        if individual:
            self.Linear_Trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(num_features)])
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(num_features)])
        else:
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)

        self.individual = individual
        self.num_features = num_features

    def forward(self, x):
        trend_init, seasonal_init = self.decomposition(x)

        if self.individual:
            list_trend_output = []
            list_seasonal_output = []

            for i in range(self.num_features):
                trend_feature = trend_init[:, :, i]
                seasonal_feature = seasonal_init[:, :, i]

                trend_output_i = self.Linear_Trend[i](trend_feature).unsqueeze(-1)
                seasonal_output_i = self.Linear_Seasonal[i](seasonal_feature).unsqueeze(-1)

                list_trend_output.append(trend_output_i)
                list_seasonal_output.append(seasonal_output_i)

            trend_output = torch.cat(list_trend_output, dim=-1)
            seasonal_output = torch.cat(list_seasonal_output, dim=-1)

        else:
            trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)

        output = trend_output + seasonal_output
        return output

