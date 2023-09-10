import torch
import random
import subprocess
import numpy as np
from tqdm import tqdm
from scipy import stats
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from utils.utils import Utils
from options.option import options
from models.scperb_model import scperb
# from dataloader.testDataset import test_dataset
from scipy import sparse
from dataloader.scperbDataset import customDataloader
import pandas as pd
import seaborn as sns

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

        condition_key = opt.condition_key

        sc.tl.rank_genes_groups(
            adata, groupby=condition_key, n_genes=100, method="wilcoxon"
        )
        diff_genes = top_100_genes
        stim = adata2numpy(adata[adata.obs[condition_key] == axis_keys["y"]])
        ctrl = adata2numpy(adata[adata.obs[condition_key] == axis_keys["x"]])
        print(adata[adata.obs[condition_key] == axis_keys["y"]])
        print(adata[adata.obs[condition_key] == axis_keys["x"]])
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
        
        # if "y1" in axis_keys.keys():
        #     real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        
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
        print(df)
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, ax = axs)
        ax.tick_params(labelsize=fontsize)
        
        # if "range" in kwargs:
        #     start, stop, step = kwargs.get("range")
        #     ax.set_xticks(np.arange(start, stop, step))
        #     ax.set_yticks(np.arange(start, stop, step))
        
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        
        # if "y1" in axis_keys.keys():
        #     if type == 'variance':
        #         y1 = np.asarray(np.var(adata2numpy(real_stim), axis=0)).ravel()
        #     else:
        #         y1 = np.asarray(np.mean(adata2numpy(real_stim), axis=0)).ravel()
        #     ax.scatter(
        #         x,
        #         y1,
        #         marker="*",
        #         c="grey",
        #         alpha=0.5,
        #         label=f"{axis_keys['x']}-{axis_keys['y1']}",
        #     )
        
        # if gene_list is not None:
        #     for i in gene_list:
        #         j = adata.var_names.tolist().index(i)
        #         x_bar = x[j]
        #         y_bar = y[j]
        #         ax.text(x_bar, y_bar, i, fontsize=11, color="black")
        #         ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
        #         if "y1" in axis_keys.keys():
        #             y1_bar = y1[j]
        #             ax.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        
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

def plot(opt, predicts):
    # utils = Utils(opt)
    
    if opt.supervise == True:
        valid = sc.read(opt.read_valid_path)

    else:
        valid = sc.read(opt.read_path)

    # valid_pred = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.ctrl_key))].copy()

    cell_type_data = valid[valid.obs[opt.cell_type_key] == opt.exclude_celltype]

    # print(adata2numpy(valid_pred))

    fig, axs = plt.subplots(2, 2, figsize = (10, 10))
    # axs[0,0].axis('off')
    # axs[0,1].axis('off')
    # axs[1,0].axis('off')
    # axs[1,1].axis('off')

    # pred[pred<5] = 5

    # pred.obs['condition'] = opt.pred_key
    pred = anndata.AnnData(predicts, obs={opt.condition_key: [opt.pred_key] * len(predicts), opt.cell_type_key: [opt.exclude_celltype] * len(predicts)}, var={"var_names": cell_type_data.var_names})

    ctrl_adata = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs['condition'] == opt.ctrl_key))]
    stim_adata = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs['condition'] == opt.stim_key))]
    # print("ctrl:")
    # print_res(adata2numpy(ctrl_adata))
    # print_res(adata2numpy(stim_adata))
    # print("stim:")
    # print_res(adata2numpy(stim_adata))
    eval_adata = ctrl_adata.concatenate(stim_adata, pred)
    eval_adata.write_h5ad(opt.model_save_path + "/result.h5ad")

    sc.tl.pca(eval_adata)
    sc.pl.pca(eval_adata, color="condition", frameon=False, ax = axs[1,0])

    CD4T = valid[valid.obs[opt.cell_type_key] == opt.exclude_celltype]

    sc.tl.rank_genes_groups(CD4T, groupby="condition", method="wilcoxon")
    diff_genes = CD4T.uns["rank_genes_groups"]["names"][opt.stim_key]
    print(diff_genes)

    conditions = {"real_stim": opt.stim_key, "pred_stim": opt.pred_key}
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

    fig.savefig(opt.save_path + '/' + opt.model_name +'_result.pdf')
    
    print("saved at:", opt.save_path + '/' + opt.model_name +'_result.pdf')

def validation(opt, model):

    opt.validation = True

    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = False, pin_memory = True)
    pred = np.empty((0, opt.input_dim))
    model.to(opt.device, non_blocking=True)
    for idx, (con, sty) in enumerate(dataloader):
        model.eval()
        eval = model.predict(con, sty)
        eval[eval<0] = 0
        # eval_new = model.predict_new(con, sty)
        pred = np.append(pred, eval, axis = 0)
    
    stim_data = dataset.get_real_stim()

    x = np.asarray(np.mean(stim_data, axis=0)).ravel()
    y = np.asarray(np.mean(pred, axis=0)).ravel()

    m, b, r_value_mean, p_value, std_err = stats.linregress(x, y)

    x = np.asarray(np.var(stim_data, axis=0)).ravel()
    y = np.asarray(np.var(pred, axis=0)).ravel()

    m, b, r_value_var, p_value, std_err = stats.linregress(x, y)

    if opt.plot == True:
        # utils = Utils(opt)
        # ctrl_data, stim_data, cell_type = dataset.get_val_data()
        # utils.plot_result(ctrl_data, stim_data, pred, cell_type, model_use)
        plot(opt, pred)

    opt.validation = False
    return (((r_value_mean ** 2) * 3) + r_value_var ** 2) * 25, r_value_mean ** 2 * 100, r_value_var ** 2 * 100

def train_model(opt, dataset):
    utils = Utils(opt)

    model = spaperb(opt)
    
    if opt.resume == True:
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
    
    model.to(opt.device, non_blocking=True)
    
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True)

    scores = -1
    best_model = 0
    pbar = tqdm(total=opt.epochs)
    for epoch in range(opt.epochs):
        loss = {}
        for idx, (con, sti, sty) in enumerate(dataloader):
            model.train()
            model.set_input(con, sti, sty)
            model.update_parameter(epoch)
            loss_dic = utils.get_loss(model)
            for (k, v) in loss_dic.items():
                if idx == 0:
                    loss[k] = v
                else:
                    loss[k] += v

        model.save(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
        tmp_scores, mean, var = validation(opt, model)
        # print("tmp:", tmp_scores)
        best_model, scores = utils.bestmodel(scores, tmp_scores, epoch, best_model, model)
        
        utils.update_pbar(loss, scores, best_model, pbar, mean, var, tmp_scores)

def fix_seed(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.seed)

def plot_graph(opt):
    model = scperb(opt)
    # print(opt.model_save_path)
    # model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_best_epoch.pt')
    model.load("./supervise_2enc_scperb_saved_one_loss/hpoly/model_Tuft" + '/'  + opt.exclude_celltype + '_best_epoch.pt')
    opt.plot = True
    validation(opt, model)

if __name__ == '__main__':
    Opt = options()
    opt = Opt.init()
    # opt.supervise = False
    # print(opt.supervise)
    # utils = Utils(opt)
    # plot(opt)
    # exit(0)
    fix_seed(opt)
    dataset = customDataloader(opt)
    
    if opt.download_data == True:
        command = "python3 DataDownloader.py"
        subprocess.call([command], shell=True)

    if opt.plot_only == True:
        opt.plot = True
        # validation(opt, model_use = opt.use_model)
        plot_graph(opt)
    
    else:
        train_model(opt, dataset)
        plot_graph(opt)
        # plot(opt)
        # opt.plot = True
        # validation(opt, model_use = opt.use_model)