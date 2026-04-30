import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import torch
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def set_seed(seed: int = 666) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_enhanced_fused_adjacency_matrix(adata_concat, adj_list, verbose=False):
    if verbose:
        print("Building fused adjacency matrix...")

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(adj_list)):
        adj_concat = scipy.linalg.block_diag(
            adj_concat, np.asarray(adj_list[batch_id].todense())
        )
    adj_spatial = sp.csr_matrix(adj_concat)

    sc.pp.pca(adata_concat, n_comps=50)
    sc.pp.neighbors(adata_concat, n_neighbors=8, use_rep="X_pca")
    sc.tl.umap(adata_concat, n_components=12, random_state=42)

    nbrs = NearestNeighbors(n_neighbors=15, metric="cosine").fit(
        adata_concat.obsm["X_umap"]
    )
    distances, indices = nbrs.kneighbors()

    adj_expr = sp.lil_matrix((adata_concat.n_obs, adata_concat.n_obs), dtype=np.float32)
    for i in range(adata_concat.n_obs):
        for j, d in zip(indices[i], 1 - distances[i]):
            if i != j and d > 0.3:
                adj_expr[i, j] = d
    adj_expr = adj_expr.tocsr()

    adj_expr = adj_expr.multiply(adj_spatial)

    def normalize_sparse(A):
        row_sums = np.array(A.sum(axis=1)).flatten() + 1e-4
        D_inv = sp.diags(1.0 / row_sums)
        return D_inv.dot(A)

    adj_sp_norm = normalize_sparse(adj_spatial)
    adj_expr_norm = normalize_sparse(adj_expr)

    alpha_list = []
    for i in range(adj_spatial.shape[0]):
        sp_neigh = adj_spatial[i].indices
        expr_neigh = adj_expr[i].indices
        if len(sp_neigh) > 5 and len(expr_neigh) > 5:
            common = len(np.intersect1d(sp_neigh, expr_neigh))
            alpha_i = common / min(len(sp_neigh), len(expr_neigh))
        else:
            alpha_i = 0.5
        alpha_list.append(alpha_i)

    alpha_arr = np.clip(np.array(alpha_list), 0.2, 0.8)

    adj_fused = sp.lil_matrix(adj_spatial.shape)
    for i in range(adj_spatial.shape[0]):
        adj_fused[i] = alpha_arr[i] * adj_sp_norm[i] + (1 - alpha_arr[i]) * adj_expr_norm[i]
    adj_fused = adj_fused.tocsr()

    rows, cols = adj_fused.nonzero()
    adata_concat.uns["edgeList"] = (rows, cols)

    if verbose:
        print(f"Fused adjacency shape: {adj_fused.shape}")
        print(f"Alpha mean: {np.mean(alpha_arr):.3f}")
        print(f"Alpha range: [{np.min(alpha_arr):.3f}, {np.max(alpha_arr):.3f}]")

    return adj_fused


def create_optimized_hard_triplets(embeddings, epoch, max_epochs, random_state=42, verbose=False):
    if verbose:
        print(f"Hard triplet mining: epoch={epoch}")

    progress = epoch / max_epochs
    n_samples = len(embeddings)

    scales = [
        max(12, n_samples // 800),
        max(22, n_samples // 500),
        max(35, n_samples // 300),
    ]

    if progress < 0.3:
        weights = [0.5, 0.3, 0.2]
    elif progress < 0.7:
        weights = [0.3, 0.4, 0.3]
    else:
        weights = [0.2, 0.3, 0.5]

    all_anchors, all_positives, all_negatives = [], [], []

    for scale_idx, (n_clusters, weight) in enumerate(zip(scales, weights)):
        if weight < 0.1:
            continue

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state + scale_idx,
            n_init=10,
            max_iter=100,
        )
        labels = kmeans.fit_predict(embeddings)

        quality = _evaluate_clustering_quality(embeddings, labels)
        target_triplets = int(2000 * weight * np.clip(quality, 0.7, 1.3))

        anchors, positives, negatives = _generate_quality_triplets(
            embeddings, labels, n_clusters, target_triplets, progress, verbose and scale_idx == 0
        )

        all_anchors.extend(anchors)
        all_positives.extend(positives)
        all_negatives.extend(negatives)

    if len(all_anchors) > 6000:
        indices = np.random.choice(len(all_anchors), 6000, replace=False)
        all_anchors = [all_anchors[i] for i in indices]
        all_positives = [all_positives[i] for i in indices]
        all_negatives = [all_negatives[i] for i in indices]

    final_n_clusters = int(np.average(scales, weights=weights))
    return np.array(all_anchors), np.array(all_positives), np.array(all_negatives), final_n_clusters


def _evaluate_clustering_quality(embeddings, labels):
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            return max(0.5, silhouette_score(embeddings, labels) + 0.5)
        return 1.0
    except Exception:
        return 1.0


def _generate_quality_triplets(embeddings, labels, n_clusters, target_triplets, progress, verbose):
    cluster_to_indices = {}
    for idx, label in enumerate(labels):
        cluster_to_indices.setdefault(label, []).append(idx)

    valid_clusters = {k: v for k, v in cluster_to_indices.items() if len(v) >= 3}
    if len(valid_clusters) < 2:
        return [], [], []

    anchors, positives, negatives = [], [], []

    cluster_centers = {
        cid: np.mean(embeddings[idxs], axis=0) for cid, idxs in valid_clusters.items()
    }
    triplets_per_cluster = max(1, target_triplets // len(valid_clusters))

    for cluster_id, cluster_nodes in valid_clusters.items():
        cluster_nodes = np.array(cluster_nodes)
        cluster_center = cluster_centers[cluster_id]

        distances_to_center = np.linalg.norm(embeddings[cluster_nodes] - cluster_center, axis=1)
        sorted_indices = np.argsort(distances_to_center)
        mid_start = len(sorted_indices) // 4
        mid_end = 3 * len(sorted_indices) // 4
        anchor_pool = cluster_nodes[sorted_indices[mid_start:mid_end]]

        num_anchors = min(triplets_per_cluster, len(anchor_pool), 100)
        if num_anchors < 1:
            continue

        if len(anchor_pool) > num_anchors:
            anchor_candidates = np.random.choice(anchor_pool, num_anchors, replace=False)
        else:
            anchor_candidates = anchor_pool

        other_clusters = [k for k in valid_clusters.keys() if k != cluster_id]
        if not other_clusters:
            continue

        for anchor in anchor_candidates:
            pos_candidates = cluster_nodes[cluster_nodes != anchor]
            if len(pos_candidates) == 0:
                continue

            anchor_emb = embeddings[anchor]
            pos_sim = np.dot(embeddings[pos_candidates], anchor_emb) / (
                np.linalg.norm(embeddings[pos_candidates], axis=1) * np.linalg.norm(anchor_emb) + 1e-8
            )
            best_pos_idx = np.argmax(pos_sim)
            positive = pos_candidates[best_pos_idx]

            negative = _select_intelligent_hard_negative(
                embeddings, anchor, other_clusters, valid_clusters, cluster_centers,
                cluster_id, progress
            )
            if negative is not None:
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

    return anchors, positives, negatives


def _select_intelligent_hard_negative(
    embeddings, anchor, other_clusters, cluster_to_indices, cluster_centers, anchor_cluster_id, progress
):
    anchor_emb = embeddings[anchor]
    neg_candidates = []
    for cluster_id in other_clusters:
        neg_candidates.extend(cluster_to_indices[cluster_id])

    if len(neg_candidates) == 0:
        return None

    neg_candidates = np.array(neg_candidates)
    neg_embeddings = embeddings[neg_candidates]
    similarities = np.dot(neg_embeddings, anchor_emb) / (
        np.linalg.norm(neg_embeddings, axis=1) * np.linalg.norm(anchor_emb) + 1e-8
    )

    sorted_indices = np.argsort(similarities)
    if progress < 0.4:
        start_idx = len(sorted_indices) // 5
        end_idx = 2 * len(sorted_indices) // 5
    elif progress < 0.8:
        start_idx = 2 * len(sorted_indices) // 5
        end_idx = 3 * len(sorted_indices) // 5
    else:
        start_idx = 3 * len(sorted_indices) // 5
        end_idx = 4 * len(sorted_indices) // 5

    if start_idx >= end_idx:
        selected_idx = len(sorted_indices) // 2
    else:
        selected_idx = sorted_indices[np.random.choice(range(start_idx, end_idx))]

    return neg_candidates[selected_idx]


def Cal_Spatial_Net(adata, rad_cutoff=150):
    assert "spatial" in adata.obsm.keys()
    coor = pd.DataFrame(adata.obsm["spatial"])
    coor.index = adata.obs.index
    coor.columns = ["imagerow", "imagecol"]

    nbrs = NearestNeighbors(radius=rad_cutoff).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    knn_list = []
    for it in range(indices.shape[0]):
        knn_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    knn_df = pd.concat(knn_list)
    knn_df.columns = ["Cell1", "Cell2", "Distance"]
    knn_df = knn_df.loc[knn_df["Distance"] > 0]

    id_cell_trans = dict(zip(range(coor.shape[0]), coor.index))
    knn_df["Cell1"] = knn_df["Cell1"].map(id_cell_trans)
    knn_df["Cell2"] = knn_df["Cell2"].map(id_cell_trans)

    adj = sp.coo_matrix(
        (np.ones(knn_df.shape[0]),
         (knn_df["Cell1"].map(dict(zip(coor.index, range(coor.shape[0])))),
          knn_df["Cell2"].map(dict(zip(coor.index, range(coor.shape[0])))))),
        shape=(coor.shape[0], coor.shape[0]),
    )
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adata.uns["adj"] = adj


def mclust_R(adata, num_cluster, used_obsm="AFPN-Net", random_seed=2020):
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        robjects.r.library("mclust")
        robjects.r["set.seed"](random_seed)
        rmclust = robjects.r["Mclust"]

        res = rmclust(adata.obsm[used_obsm], num_cluster, "EEE")
        mclust_res = np.array(res[-2])

        adata.obs["mclust"] = mclust_res
        adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")
        return adata
    except Exception as e:
        kmeans = KMeans(n_clusters=num_cluster, random_state=random_seed, n_init=10)
        adata.obs["mclust"] = kmeans.fit_predict(adata.obsm[used_obsm])
        adata.obs["mclust"] = adata.obs["mclust"].astype("category")
        return adata


def match_cluster_labels(true_labels, pred_labels):
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix

        true_labels_int = pd.Categorical(true_labels).codes
        pred_labels_int = pd.Categorical(pred_labels).codes

        cm = confusion_matrix(true_labels_int, pred_labels_int)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = dict(zip(col_ind, row_ind))

        matched_labels = [mapping.get(label, label) for label in pred_labels_int]
        unique_true = pd.Categorical(true_labels).categories
        matched_labels = [
            unique_true[label] if label < len(unique_true) else f"cluster_{label}"
            for label in matched_labels
        ]
        return matched_labels
    except Exception:
        return pred_labels
