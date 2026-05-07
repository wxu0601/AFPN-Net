import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class Encoder(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, dropout):
        super().__init__()
        self.gat1 = GATConv(
            in_dim, num_hidden, heads=1, concat=False,
            dropout=dropout, add_self_loops=False
        )
        self.gat2 = GATConv(
            num_hidden, out_dim, heads=1, concat=False,
            dropout=dropout, add_self_loops=False
        )
        self.gcn1 = GCNConv(in_dim, num_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(num_hidden, out_dim, add_self_loops=False)

    def forward(self, features, edge_index):
        h_gat = F.elu(self.gat1(features, edge_index))
        h_gat = F.dropout(h_gat, p=0.2, training=self.training)
        h_gat = self.gat2(h_gat, edge_index)

        h_gcn = F.relu(self.gcn1(features, edge_index))
        h_gcn = F.dropout(h_gcn, p=0.2, training=self.training)
        h_gcn = self.gcn2(h_gcn, edge_index)

        h = torch.cat([h_gat, h_gcn], dim=1)
        return h


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)


class AFPNNet(nn.Module):
    def __init__(self, hidden_dims, dropout=0.2, mask_rate=0.3, use_discriminator=True):
        super().__init__()
        in_dim, num_hidden, out_dim = hidden_dims
        self.encoder = Encoder(in_dim, num_hidden, out_dim, dropout)
        self.decoder = Decoder(out_dim * 2, in_dim, dropout)
        self.use_discriminator = use_discriminator
        if self.use_discriminator:
            self.discriminator = Discriminator(in_features=out_dim * 2)
        self.mask_rate = mask_rate

    def mask_by_nodes_(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        return out_x, mask_nodes

    def forward(self, features, edge_index):
        masked_features, _ = self.mask_by_nodes_(features, self.mask_rate)
        encoded_features = self.encoder(masked_features, edge_index)
        reconstructed_features = self.decoder(encoded_features)
        if self.use_discriminator:
            disc_out = self.discriminator(encoded_features)
            return reconstructed_features, encoded_features, disc_out
        return reconstructed_features, encoded_features, None


class ConditionalAE(nn.Module):
    def __init__(self, in_dim, num_batches, batch_emb_dim=8, latent_dim=64, hidden_dim=64):
        super().__init__()
        self.batch_embedding = nn.Embedding(num_batches, batch_emb_dim)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim + batch_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + batch_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x, batch_idx):
        b = self.batch_embedding(batch_idx)
        h = torch.cat([x, b], dim=1)
        z = self.encoder(h)
        return z

    def decode(self, z, batch_idx):
        b = self.batch_embedding(batch_idx)
        h = torch.cat([z, b], dim=1)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x, batch_idx):
        z = self.encode(x, batch_idx)
        x_hat = self.decode(z, batch_idx)
        return x_hat, z
