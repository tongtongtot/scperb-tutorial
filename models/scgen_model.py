import os
import pdb
import torch
import numpy as np
from torch import nn
from scipy import stats
from scipy import sparse
from models.scgen_vae import SCGENVAE
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

class SCGEN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.criterion = nn.MSELoss(reduction = 'none')
        self.model = SCGENVAE(opt).to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        self.loss_stat = {}

    def set_input(self, x):
        self.x = x
        
    def forward(self):
        self.model.train()
        self.out = self.model(self.x)
        return self.out

    def compute_loss_sig(self):
        z, qz_m, qz_v = self.out['z']
        if self.opt.alpha != 0:
            kld = kl(Normal(qz_m, torch.sqrt(torch.exp(qz_v))),Normal(0, 1)).sum(dim=1)
        else: 
            kld = torch.tensor(0)
        rl = self.get_reconstruction_loss(self.out['output'], self.x)
        loss = torch.mean(0.5 * rl + 0.5 * (kld * self.opt.alpha))
        self.loss = loss
        self.loss_stat = {
            'kl_loss': torch.mean(kld * self.opt.alpha),
            'recon_loss': rl.mean(),
            'total_loss': self.loss
        }
    
    def get_reconstruction_loss(self, x, px):
        loss = ((x - px) ** 2).sum(dim=1)
        return loss

    def compute_loss_dou(self):
        con = self.x[0].to(self.opt.device)
        sti = self.x[1].to(self.opt.device)
        
        z_con, m_con, v_con = self.out['z_con']
        z_sti, m_sti, v_sti = self.out['z_sti']
        
        kld_con = kl(Normal(m_con, torch.sqrt(torch.exp(v_con))),Normal(0, 1)).sum(dim=1)
        kld_sti = kl(Normal(m_sti, torch.sqrt(torch.exp(v_sti))),Normal(0, 1)).sum(dim=1)

        rl_con = self.criterion(self.out['output'][0], con).sum(dim = 1)
        rl_sti = self.criterion(self.out['output'][1], sti).sum(dim = 1)
        rl_cng = self.criterion(self.out['output'][0], sti).sum(dim = 1)
        
        loss = torch.mean(0.5 * (rl_con + rl_sti) + 0.5 * ((kld_con + kld_sti) * self.opt.alpha) + rl_cng * self.opt.beta)
        self.loss = loss
        self.loss_stat = {
            'kl_con': torch.mean(kld_con * self.opt.alpha),
            'kl_sti': torch.mean(kld_sti * self.opt.alpha),
            'rl_con': rl_con.mean(),
            'rl_sti': rl_sti.mean(),
            'rl_cng': rl_cng.mean() * self.opt.beta,
            'total_loss': self.loss,
            'kl_loss': torch.mean(0.5 * ((kld_con + kld_sti) * self.opt.alpha)),
            'recon_loss': torch.mean(0.5 * (rl_con + rl_sti)),
        }

    def get_current_loss(self):
        return self.loss_stat

    def get_loss(self):
        return self.loss

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def update_parameter(self):
        self.forward()
        if self.opt.model_use == 'vae_test1':
            self.compute_loss_dou()
        else:
            self.compute_loss_sig()
        self.backward()
    
    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).to(self.opt.device)
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

    def _avg_vector(self, con, sti):
        latent_con, _, __= self.model.get_z(self.adata2tensor(con))
        latent_sti, _, __= self.model.get_z(self.adata2tensor(sti))
        latent_con_avg = np.average(self.tensor2numpy(latent_con), axis=0)
        latent_sti_avg = np.average(self.tensor2numpy(latent_sti), axis=0)
        return self.numpy2tensor((latent_sti_avg - latent_con_avg))
    
    def get_stim_pred(self, ctrl_x, stim_x, pred_data):
        delta = self._avg_vector(ctrl_x, stim_x)
        z, _, __ = self.model.get_z(self.adata2tensor(pred_data))
        stim_pred = delta + z
        return self.model.decoder(stim_pred)

    def predict(self, pred_data, ctrl_data, stim_data):
        self.model.eval()
        gen_img = self.get_stim_pred(ctrl_data, stim_data, pred_data)
        return self.tensor2numpy(gen_img)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)