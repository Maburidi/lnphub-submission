from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, GINEConv, global_mean_pool


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.atomic = nn.Embedding(119, emb_dim)
        self.degree = nn.Embedding(6, emb_dim)
        self.charge = nn.Embedding(11, emb_dim)
        self.aromatic = nn.Embedding(2, emb_dim)

    def forward(self, x):
        return self.atomic(x[:, 0]) + self.degree(x[:, 1]) + self.charge(x[:, 2]) + self.aromatic(x[:, 3])


class BondEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.bond_type = nn.Embedding(4, emb_dim)
        self.conjugated = nn.Embedding(2, emb_dim)
        self.ring = nn.Embedding(2, emb_dim)

    def forward(self, edge_attr):
        return self.bond_type(edge_attr[:, 0]) + self.conjugated(edge_attr[:, 1]) + self.ring(edge_attr[:, 2])


class GCNRegressor(nn.Module):
    def __init__(self, emb_dim: int = 128, layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.atom = AtomEncoder(emb_dim)
        self.convs = nn.ModuleList([GCNConv(emb_dim, emb_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_dim, 1))

    def forward(self, data):
        h = self.atom(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(torch.relu(norm(conv(h, data.edge_index))))
        return self.head(global_mean_pool(h, data.batch)).view(-1)


class GINRegressor(nn.Module):
    def __init__(self, emb_dim: int = 128, layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.atom = AtomEncoder(emb_dim)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            self.convs.append(GINConv(mlp))
        self.norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_dim, 1))

    def forward(self, data):
        h = self.atom(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(torch.relu(norm(conv(h, data.edge_index))))
        return self.head(global_mean_pool(h, data.batch)).view(-1)


class DMPNNRegressor(nn.Module):
    """Edge-aware message passing baseline used as a D-MPNN-style oracle."""

    def __init__(self, emb_dim: int = 128, layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.atom = AtomEncoder(emb_dim)
        self.bond = BondEncoder(emb_dim)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            self.convs.append(GINEConv(mlp))
        self.norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_dim, 1))

    def forward(self, data):
        h = self.atom(data.x)
        e = self.bond(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(torch.relu(norm(conv(h, data.edge_index, e))))
        return self.head(global_mean_pool(h, data.batch)).view(-1)
