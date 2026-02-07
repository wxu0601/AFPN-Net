import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .models import AFPNNet, ConditionalAE
from .utils import create_optimized_hard_triplets


def train_afpnnet_fused_innovation(
    adata,
    hidden_dims=(512, 32),
    n_epochs=500,
    lr=0.001,
    key_added="AFPN-Net",
    gradient_clipping=5,
    weight_decay=0.0001,
    margin=0.5,
    verbose=False,
    random_seed=666,
    device=None,
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    edge_list = adata.uns["edgeList"]
    edge_index = torch.LongTensor(np.array([edge_list[0], edge_list[1]]))

    x = adata.X.todense() if sp.issparse(adata.X) else adata.X
    data = Data(edge_index=edge_index, x=torch.FloatTensor(x)).to(device)

    in_dim = data.x.shape[1]
    hidden_dim = int(hidden_dims[0])
    out_dim = int(hidden_dims[1])

    model = AFPNNet(hidden_dims=[in_dim, hidden_dim, out_dim], dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if verbose:
        print(f"Model: {in_dim} -> {hidden_dim} -> {out_dim}")
        print(f"Data: x={tuple(data.x.shape)}, edge_index={tuple(data.edge_index.shape)}")

    if verbose:
        print("Pretrain stage (reconstruction only)...")
    for epoch in tqdm(range(0, 300), disable=not verbose):
        model.train()
        optimizer.zero_grad()
        out, z, disc_out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    with torch.no_grad():
        _, z, _ = model(data.x, data.edge_index)
    adata.obsm["pretrain"] = z.cpu().detach().numpy()

    if verbose:
        print("Main training stage (recon + triplet + discriminator)...")

    anchor_ind, positive_ind, negative_ind = [], [], []
    current_n_clusters = 30

    for epoch in tqdm(range(300, n_epochs), disable=not verbose):
        if epoch % 80 == 0 or epoch == 300:
            adata.obsm["pretrain"] = z.cpu().detach().numpy()
            embeddings = adata.obsm["pretrain"]

            anchor_ind, positive_ind, negative_ind, optimal_clusters = create_optimized_hard_triplets(
                embeddings,
                epoch - 300,
                n_epochs - 300,
                random_state=random_seed + epoch,
                verbose=False,
            )
            current_n_clusters = optimal_clusters

        model.train()
        optimizer.zero_grad()
        out, z, disc_out = model(data.x, data.edge_index)
        mse_loss = F.mse_loss(data.x, out)

        if len(anchor_ind) > 0:
            anchor_arr = z[anchor_ind, :]
            positive_arr = z[positive_ind, :]
            negative_arr = z[negative_ind, :]

            triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")
            triplet_loss = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)

            disc_loss_fn = torch.nn.BCEWithLogitsLoss()
            discriminator_loss = disc_loss_fn(disc_out, torch.ones_like(disc_out))

            alpha = 0.65
            beta = 1.10
            gamma = 0.12

            loss = alpha * mse_loss + beta * triplet_loss + gamma * discriminator_loss
        else:
            loss = mse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, z, _ = model(data.x, data.edge_index)
    adata.obsm[key_added] = z.cpu().detach().numpy()

    if verbose:
        print(f"Done. Suggested clusters: {current_n_clusters}")

    return adata


def compute_mnn_pairs(X, batch_codes, k=5, max_pairs=5000, verbose=False):
    unique_batches = np.unique(batch_codes)
    pairs = []

    for i, b1 in enumerate(unique_batches):
        for b2 in unique_batches[i + 1:]:
            idx1 = np.where(batch_codes == b1)[0]
            idx2 = np.where(batch_codes == b2)[0]
            if len(idx1) < 2 or len(idx2) < 2:
                continue

            n1 = min(len(idx1), 1000)
            n2 = min(len(idx2), 1000)
            sample1 = np.random.choice(idx1, n1, replace=False)
            sample2 = np.random.choice(idx2, n2, replace=False)

            X1 = X[sample1]
            X2 = X[sample2]

            knn_12 = NearestNeighbors(n_neighbors=min(k, n2)).fit(X2)
            dist12, ind12 = knn_12.kneighbors(X1)

            knn_21 = NearestNeighbors(n_neighbors=min(k, n1)).fit(X1)
            dist21, ind21 = knn_21.kneighbors(X2)

            for local_i, global_i in enumerate(sample1):
                for j_pos in range(ind12.shape[1]):
                    local_j = ind12[local_i, j_pos]
                    global_j = sample2[local_j]

                    neighbors_back_local = ind21[local_j]
                    neighbors_back_global = sample1[neighbors_back_local]
                    if global_i in neighbors_back_global:
                        pairs.append((global_i, global_j))
                        if len(pairs) >= max_pairs:
                            break
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    if verbose:
        print(f"MNN pairs: {len(pairs)}")

    if len(pairs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    anchor_idx = np.array([p[0] for p in pairs], dtype=int)
    pos_idx = np.array([p[1] for p in pairs], dtype=int)
    return anchor_idx, pos_idx


def train_conditional_ae_on_embedding(
    adata,
    key_in="AFPN-Net",
    key_out="AFPN-Net_CAE",
    batch_key="batch_name",
    latent_dim=32,
    batch_emb_dim=8,
    hidden_dim=64,
    n_epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    k_neighbors=15,
    mnn_k=5,
    lambda_triplet=1.0,
    lambda_neigh=0.5,
    margin=0.5,
    device=None,
    verbose=False,
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X = adata.obsm[key_in].astype(np.float32)
    N, D = X.shape

    batch_cat = adata.obs[batch_key].astype("category")
    batch_codes = batch_cat.cat.codes.values
    num_batches = len(batch_cat.cat.categories)

    if verbose:
        print(f"CAE: N={N}, D={D}, batches={num_batches}")

    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn_model.fit(X)
    dist, idx = nn_model.kneighbors(X)
    neighbor_indices = idx[:, 1:]
    neighbor_distances = dist[:, 1:]

    neighbor_indices_t = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    neighbor_distances_t = torch.tensor(neighbor_distances, dtype=torch.float32, device=device)

    anchor_idx_np, pos_idx_np = compute_mnn_pairs(
        X, batch_codes, k=mnn_k, max_pairs=5000, verbose=verbose
    )

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    batch_t = torch.tensor(batch_codes, dtype=torch.long, device=device)

    cae = ConditionalAE(
        in_dim=D,
        num_batches=num_batches,
        batch_emb_dim=batch_emb_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optim = torch.optim.Adam(cae.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    for epoch in range(n_epochs):
        cae.train()
        optim.zero_grad()

        x_hat, z = cae(X_t, batch_t)
        recon_loss = F.mse_loss(x_hat, X_t)

        anchors_z = z.unsqueeze(1)
        neighbors_z = z[neighbor_indices_t]
        latent_distances = torch.norm(anchors_z - neighbors_z, dim=2)
        neigh_loss = ((latent_distances - neighbor_distances_t) ** 2).mean()

        if len(anchor_idx_np) > 0:
            neg_idx_np = []
            for a in anchor_idx_np:
                b = batch_codes[a]
                cand = np.where(batch_codes != b)[0]
                neg_idx_np.append(np.random.choice(cand))
            neg_idx_np = np.array(neg_idx_np, dtype=int)

            anchor_idx_t = torch.tensor(anchor_idx_np, dtype=torch.long, device=device)
            pos_idx_t = torch.tensor(pos_idx_np, dtype=torch.long, device=device)
            neg_idx_t = torch.tensor(neg_idx_np, dtype=torch.long, device=device)

            z_anchor = z[anchor_idx_t]
            z_pos = z[pos_idx_t]
            z_neg = z[neg_idx_t]
            triplet_loss = triplet_loss_fn(z_anchor, z_pos, z_neg)
        else:
            triplet_loss = torch.tensor(0.0, device=device)

        loss = recon_loss + lambda_neigh * neigh_loss

        loss.backward()
        optim.step()

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(
                f"CAE epoch {epoch:03d}: loss={loss.item():.4f} "
                f"recon={recon_loss.item():.4f} neigh={neigh_loss.item():.4f}"
            )

    cae.eval()
    with torch.no_grad():
        _, z_final = cae(X_t, batch_t)
    adata.obsm[key_out] = z_final.cpu().numpy()

    if verbose:
        print(f"CAE done: {adata.obsm[key_out].shape}")

    return adata
