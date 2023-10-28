import os
# import torch
import random
import subprocess
import numpy as np
from tqdm import tqdm
from scipy import stats
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
# from dataloader.testDataset import test_dataset
from scipy import sparse
import pandas as pd
import seaborn as sns
import scgen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exclude_cell", type = str, default = 'CD4T')
parser.add_argument("--dataset", type = str, default = 'pbmc')
parser.add_argument("--cell_type_key", type = str, default = 'cell_type')
parser.add_argument("--ctrl_key", type = str, default = 'control')
parser.add_argument("--stim_key", type = str, default = 'stimulated')
opt = parser.parse_args()

if opt.dataset == 'hpoly':
    opt.stim_key = 'Hpoly.Day10'
    opt.ctrl_key = 'Control'
    opt.cell_type_key = 'cell_label'

def adata2numpy(adata):
    if sparse.issparse(adata.X):
        return adata.X.A
    else:
        return adata.X

def print_res(stim):
    a = stim.mean()
    var = stim.var()
    b = stim.max()
    c = stim.min()
    mid = np.median(stim)

    print("mean:", a)
    print("max:", b)
    print("min:", c)
    print("var:", var)
    print("median", mid)

def reg_plot(
        axs,
        adata,
        axis_keys,
        labels,
        gene_list=None,
        top_100_genes=None,
        show=False,
        legend=True,
        title=None,
        verbose=False,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        type='mean',
        **kwargs,
    ):

        sns.set()
        sns.set(color_codes=True)

        condition_key = 'condition'
        sc.tl.rank_genes_groups(
            adata, groupby=condition_key, n_genes=100, method="wilcoxon"
        )
        diff_genes = top_100_genes
        stim = adata2numpy(adata[adata.obs[condition_key] == axis_keys["y"]])
        ctrl = adata2numpy(adata[adata.obs[condition_key] == axis_keys["x"]])
        
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]])
            ctrl_diff = adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]])

            if type == 'variance':

                x_diff = np.asarray(np.var(ctrl_diff, axis=0)).ravel()
                y_diff = np.asarray(np.var(stim_diff, axis=0)).ravel()
            else: 
                x_diff = np.asarray(np.mean(ctrl_diff, axis=0)).ravel()
                y_diff = np.asarray(np.mean(stim_diff, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        
        if type == 'variance':
            x = np.asarray(np.var(ctrl, axis=0)).ravel()
            y = np.asarray(np.var(stim, axis=0)).ravel()
        else:
            x = np.asarray(np.mean(ctrl, axis=0)).ravel()
            y = np.asarray(np.mean(stim, axis=0)).ravel()
        
        m, b, r_value, p_value, std_err = stats.linregress(x, y)

        if verbose:
            print("All genes var: ", r_value**2)
        
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, ax = axs)
        ax.tick_params(labelsize=fontsize)
        
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        
        if "y1" in axis_keys.keys():
            if type == 'variance':
                y1 = np.asarray(np.var(adata2numpy(real_stim), axis=0)).ravel()
            else:
                y1 = np.asarray(np.mean(adata2numpy(real_stim), axis=0)).ravel()
            ax.scatter(
                x,
                y1,
                marker="*",
                c="grey",
                alpha=0.5,
                label=f"{axis_keys['x']}-{axis_keys['y1']}",
            )
        
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                ax.text(x_bar, y_bar, i, fontsize=11, color="black")
                ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    ax.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        
        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        if title is None:
            ax.set_title("", fontsize=12)
        
        else:
            ax.set_title(title, fontsize=12)
        
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.4f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.4f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        return r_value**2

def plot(pred):
    exclude_celltype = opt.exclude_cell
    cell_type_key = opt.cell_type_key
    ctrl_key = opt.ctrl_key
    stim_key = opt.stim_key

    fig, axs = plt.subplots(2, 2, figsize = (10, 10))
    eval_adata = pred

    sc.tl.pca(eval_adata)
    sc.pl.pca(eval_adata, color="condition", frameon=False, ax = axs[1,0])

    CD4T = valid[valid.obs[cell_type_key] == exclude_celltype]

    sc.tl.rank_genes_groups(CD4T, groupby="condition", method="wilcoxon")
    diff_genes = CD4T.uns["rank_genes_groups"]["names"][stim_key]
    print(diff_genes)

    conditions = {"real_stim": stim_key, "pred_stim": 'pred'}
    mean_labels = {"x": "ctrl mean", "y": "stim mean"}
    var_labels = {"x": "ctrl var", "y": "stim var"}

    reg_plot(     
        axs = axs[0,0],
        adata=eval_adata, 
        axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
        gene_list=diff_genes[:5],
        top_100_genes=diff_genes[:100],
        legend=False,
        title="",
        labels=mean_labels,
        fontsize=10,
        show=True,
        type = 'mean',
    )
        
    reg_plot(
        axs = axs[0,1],
        adata=eval_adata, 
        axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
        gene_list=diff_genes[:5],
        top_100_genes=diff_genes[:100],
        legend=False,
        labels=var_labels,
        title="",
        fontsize=10,
        type = 'variance',
        show=False
    )

    sc.pl.violin(eval_adata, keys=diff_genes[0], groupby="condition", ax = axs[1,1])
    print(diff_genes[0])

    fig.savefig('result.pdf')
    
    print('saved at:result.pdf')

train = sc.read(f"../data/train_{opt.dataset}.h5ad")
valid = sc.read(f"../data/valid_{opt.dataset}.h5ad")
# train = train[~((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "stimulated"))]
# valid = valid[~((valid.obs["cell_type"] == "CD4T") & (valid.obs["condition"] == "stimulated"))]
z_dim = 20
network = scgen.CVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.1, model_path="../models/CVAE/pbmc/all/models/scgen")
# network.cuda()
network.train(train, use_validation=True, valid_data=valid, n_epochs=100)
labels, _ = scgen.label_encoder(train)
# train = sc.read("../data/train_pbmc.h5ad")
CD4T = valid[valid.obs[opt.cell_type_key] == opt.exclude_cell]
# unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
unperturbed_data = valid[((valid.obs[opt.cell_type_key] == opt.exclude_cell) & (valid.obs["condition"] == opt.ctrl_key))]
fake_labels = np.ones((len(unperturbed_data), 1))
predicted_cells = network.predict(unperturbed_data, fake_labels)
print("predict_size", predicted_cells.shape)
adata = sc.AnnData(predicted_cells, obs={"condition": ["pred"]*len(fake_labels)})
adata.var_names = CD4T.var_names
all_adata = CD4T.concatenate(adata)
os.makedirs(f"../data/reconstructed/CVAE/{opt.dataset}", exist_ok=True)
all_adata.write(f"../data/reconstructed/CVAE/{opt.dataset}/CVAE_{opt.exclude_cell}.h5ad")
print("saved at:" + f"../data/reconstructed/CVAE/{opt.dataset}/CVAE_{opt.exclude_cell}.h5ad")
# plot(all_adata)