# from options.opt import opt
import torch
import random
import numpy as np
import scanpy as sc
from scipy import sparse
import torch.utils.data as data

class originalDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        train = sc.read(self.opt.read_path)
        if not opt.supervise:
            train = train[~((train.obs[opt.cell_type_key] == opt.exclude_celltype) & (train.obs[opt.condition_key] == opt.stim_key))]
            print("check")

        # sc.pp.log1p(train) 
        valid = sc.read(self.opt.read_valid_path)
        # sc.pp.log1p(valid)
        self.return_valid = valid
        self.cell_type = valid[valid.obs[opt.cell_type_key] == opt.exclude_celltype]
        stim = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.stim_key))]
        # self.valid = valid[~((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs[opt.condition_key] == opt.stim_key))]
        self.pred_data = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.ctrl_key))]
        self.stim = self.adata2numpy(stim)
        self.ctrl_data = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.ctrl_key))]
        self.stim_data = valid[((valid.obs[opt.cell_type_key] == opt.exclude_celltype) & (valid.obs["condition"] == opt.stim_key))]
        self.stim_np = self.adata2numpy(self.stim_data)
        self.len_valid = len(self.pred_data)
        
        con, sti = self.balance(train)
        self.len = len(sti)
        self.sti_np = self.adata2tensor(sti)
        self.con_np = self.adata2tensor(con)
        self.style = torch.ones(self.opt.input_dim)

    def get_real_stim(self):
        return self.stim_np

    def get_stat(self):
        con_num = self.adata2numpy(self.con)
        sti_num = self.adata2numpy(self.sti)
        con_data = con_num[con_num > 0]
        sti_data = sti_num[sti_num > 0]
        return float(con_data.mean()), float(con_data.var()), float(sti_data.mean()), float(sti_data.var())

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data)
        else:
            Exception("This is not a numpy")
        return data

    def tensor2numpy(self, data):
        data = data.cpu().detach().numpy()
        return data

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X
    
    def adata2tensor(self, adata):
        return self.numpy2tensor(self.adata2numpy(adata))
    
    def __getitem__(self, idx):
        return (self.con_np[idx], self.sti_np[idx], self.style)

    def __len__(self):
        return self.len

    def get_val_data(self):
        return self.pred_data, self.ctrl_data, self.stim_data, self.stim, self.cell_type

    def balance(self, adata):
        ctrl = adata[adata.obs['condition'] == self.opt.ctrl_key]
        stim = adata[adata.obs['condition'] == self.opt.stim_key]
        ctrl_cell_type = ctrl.obs[self.opt.cell_type_key]
        stim_cell_type = stim.obs[self.opt.cell_type_key]
        class_num = np.unique(ctrl_cell_type)
        max_num = {}
        
        for i in class_num:

            max_num[i] = (max(ctrl_cell_type[ctrl_cell_type == i].shape[0], stim_cell_type[stim_cell_type == i].shape[0]))
        
        ctrl_index_add = []
        strl_index_add = []

        for i in class_num:
            ctrl_class_index = np.array(ctrl_cell_type == i)
            stim_class_index = np.array(stim_cell_type == i)
            stim_fake = np.ones(len(stim_cell_type))
            
            ctrl_index_cls = np.nonzero(ctrl_class_index)[0]
            stim_index_cls = np.nonzero(stim_class_index)[0]
            stim_fake = np.nonzero(stim_fake)[0]
            stim_len = len(stim_index_cls)

            if stim_len == 0:
                stim_len = len(stim_cell_type)
                ctrl_index_cls = ctrl_index_cls[np.random.choice(len(ctrl_index_cls), max_num[i])]
                stim_index_cls = stim_fake[np.random.choice(stim_len, max_num[i])]
            
            else:
                ctrl_index_cls = ctrl_index_cls[np.random.choice(len(ctrl_index_cls), max_num[i])]
                stim_index_cls = stim_index_cls[np.random.choice(stim_len, max_num[i])]
                
            ctrl_index_add.append(ctrl_index_cls)
            strl_index_add.append(stim_index_cls)

        balanced_data_ctrl = ctrl[np.concatenate(ctrl_index_add)]
        balanced_data_stim = stim[np.concatenate(strl_index_add)]

        return balanced_data_ctrl, balanced_data_stim