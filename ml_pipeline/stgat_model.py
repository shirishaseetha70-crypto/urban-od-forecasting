import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class STGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.3):
        super(STGAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout, concat=False)
        self.temporal_conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: [num_nodes, seq_len, in_channels]
        seq_len = x.shape[1]

        spatial_out = []
        for t in range(seq_len):
            xt = x[:, t, :]
            xt = self.relu(self.gat1(xt, edge_index))
            xt = self.dropout(xt)
            xt = self.relu(self.gat2(xt, edge_index))
            spatial_out.append(xt)

        # Stack: [num_nodes, hidden, seq_len]
        spatial_out = torch.stack(spatial_out, dim=2)

        # Temporal convolutions
        temp = self.relu(self.batch_norm1(self.temporal_conv1(spatial_out)))
        temp = self.relu(self.batch_norm2(self.temporal_conv2(temp)))

        # Take last timestep
        out = temp[:, :, -1]
        return self.fc_out(out)
