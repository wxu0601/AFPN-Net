import os
import sys
import warnings

# allow running this file directly: python afpn_net/run_dlpfc.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari_score, normalized_mutual_info_score

from afpn_net import (
    set_seed,
    get_device,
    Cal_Spatial_Net,
    build_enhanced_fused_adjacency_matrix,
    train_afpnnet_fused_innovation,
    train_conditional_ae_on_embedding,
    mclust_R,
    match_cluster_labels,
)


def main():
    set_seed(666)
    device = get_device()
    print(f"Device: {device}")

    data_root = "D:/000backup project/afpn_net/DLPFC"
    section_ids = ["151673", "151674", "151675", "151676"]

    batch_list = []
    adj_list = []

    for section_id in section_ids:
        input_dir = os.path.join(data_root, section_id)

        adata = sc.read_visium(
            path=input_dir,
            count_file=section_id + "_filtered_feature_bc_matrix.h5",
            load_images=True,
        )
        adata.var_names_make_unique(join="++")

        ann_df = pd.read_csv(
            os.path.join(input_dir, section_id + "_truth.txt"),
            sep="\t",
            header=None,
            index_col=0,
        )
        ann_df.columns = ["Ground Truth"]
        ann_df[ann_df.isna()] = "unknown"
        adata.obs["Ground Truth"] = ann_df.loc[adata.obs_names, "Ground Truth"].astype("category")

        adata.obs_names = [x + "_" + section_id for x in adata.obs_names]
        Cal_Spatial_Net(adata, rad_cutoff=150)

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var["highly_variable"]]

        adj_list.append(adata.uns["adj"])
        batch_list.append(adata)

        print(f"Loaded {section_id}: {adata.shape}")

    adata_concat = ad.concat(batch_list, label="slice_name", keys=section_ids)
    adata_concat.obs["Ground Truth"] = adata_concat.obs["Ground Truth"].astype("category")
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype("category")
    print(f"Concat: {adata_concat.shape}")

    build_enhanced_fused_adjacency_matrix(adata_concat, adj_list, verbose=True)

    adata_concat = train_afpnnet_fused_innovation(
        adata_concat,
        verbose=True,
        device=device,
        margin=0.5,
        key_added="AFPN-Net",
    )

    adata_concat = train_conditional_ae_on_embedding(
        adata_concat,
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
        device=device,
        verbose=True,
    )

    adata_concat = mclust_R(adata_concat, num_cluster=7, used_obsm="AFPN-Net_CAE")
    adata_eval = adata_concat[adata_concat.obs["Ground Truth"] != "unknown"].copy()

    overall_ari = ari_score(adata_eval.obs["Ground Truth"], adata_eval.obs["mclust"])
    overall_nmi = normalized_mutual_info_score(adata_eval.obs["Ground Truth"], adata_eval.obs["mclust"])
    print(f"Overall ARI: {overall_ari:.3f}")
    print(f"Overall NMI: {overall_nmi:.3f}")

    sc.pp.neighbors(adata_eval, use_rep="AFPN-Net_CAE", random_state=666)
    sc.tl.umap(adata_eval, random_state=666)

    adata_eval.obs["mclust"] = pd.Series(
        match_cluster_labels(adata_eval.obs["Ground Truth"], adata_eval.obs["mclust"].values),
        index=adata_eval.obs.index,
        dtype="category",
    )

    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.size"] = 12

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.4)

    sc.pl.umap(adata_eval, color="batch_name", ax=axes[0], show=False, frameon=True, title="batch_name")
    sc.pl.umap(adata_eval, color="Ground Truth", ax=axes[1], show=False, frameon=True, title="Ground Truth")
    sc.pl.umap(adata_eval, color="mclust", ax=axes[2], show=False, frameon=True, title="mclust")

    plt.tight_layout()
    plt.show()

    per_slice = []
    for section_id in section_ids:
        per_slice.append(adata_eval[adata_eval.obs["batch_name"] == section_id])

    ari_list, nmi_list = [], []
    for i, adata_slice in enumerate(per_slice):
        a = ari_score(adata_slice.obs["Ground Truth"], adata_slice.obs["mclust"])
        n = normalized_mutual_info_score(adata_slice.obs["Ground Truth"], adata_slice.obs["mclust"])
        ari_list.append(round(a, 3))
        nmi_list.append(round(n, 3))
        print(f"{section_ids[i]}: ARI={a:.3f}, NMI={n:.3f}")

    fig, ax = plt.subplots(1, len(per_slice), figsize=(4 * len(per_slice), 4), gridspec_kw={"wspace": 0.1})
    if len(per_slice) == 1:
        ax = [ax]

    for i, adata_slice in enumerate(per_slice):
        sc.pl.spatial(
            adata_slice,
            img_key=None,
            color=["mclust"],
            title=[f"{section_ids[i]}\nNMI={nmi_list[i]}, ARI={ari_list[i]}"],
            legend_loc=None if i < len(per_slice) - 1 else "right margin",
            legend_fontsize=10,
            show=False,
            ax=ax[i],
            frameon=False,
            spot_size=200,
        )

    plt.tight_layout()
    plt.show()

    print("Summary:")
    print(f"Per-slice ARI: {ari_list}")
    print(f"Per-slice NMI: {nmi_list}")
    print(f"Mean ARI: {np.mean(ari_list):.3f}")
    print(f"Mean NMI: {np.mean(nmi_list):.3f}")


if __name__ == "__main__":
    import os

    print("PID:", os.getpid())

    main()
