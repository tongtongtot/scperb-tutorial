import torch
import numpy as np
import torch.nn as nn
from scipy import stats
from scipy import sparse
from models.origin_vae import VAE
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

class original_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = VAE(opt)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), opt.lr)
        self.cirterion = nn.MSELoss(reduction = 'mean')
        self.cos = torch.nn.CosineSimilarity()
        self.loss_stat = {"total_loss": 1e100,}
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def compute_loss(self):

        con_m_con = self.out['mu_con_con']
        con_m_sti = self.out['mu_sti_con']
        con_v_con = self.out['var_con_con']
        con_v_sti = self.out['var_sti_con']
        sty_m_con = self.out['mu_con_con']
        sty_m_sti = self.out['mu_sti_con']
        sty_v_con = self.out['var_con_con']
        sty_v_sti = self.out['var_sti_con']

        recon_loss_real1 = self.cirterion(self.out['real_sti_recon'], self.sti)
        recon_loss_fake1 = self.cirterion(self.out['fake_sti_recon'], self.sti)
        recon_loss_real2 = self.cirterion(self.out['real_con_recon'], self.con)
        recon_loss_fake2 = self.cirterion(self.out['fake_con_recon'], self.con)

        con_kld_con = kl(Normal(con_m_con, torch.sqrt(torch.exp(con_v_con))),Normal(0, 1)).mean()
        con_kld_sti = kl(Normal(con_m_sti, torch.sqrt(torch.exp(con_v_sti))),Normal(0, 1)).mean()
        sty_kld_con = kl(Normal(sty_m_con, torch.sqrt(torch.exp(sty_v_con))),Normal(0, 1)).mean()
        sty_kld_sti = kl(Normal(sty_m_sti, torch.sqrt(torch.exp(sty_v_sti))),Normal(0, 1)).mean()

        # delta_loss = self.cirterion(self.out['delta'], self.out['real_delta'])

        self.loss = 200 * (recon_loss_real1 + recon_loss_real2 + recon_loss_fake1 + recon_loss_fake2)  + con_kld_con + con_kld_sti + sty_kld_con + sty_kld_sti
        
        self.loss_stat = {
            "total_loss": self.loss,
            'recon': 200 * (recon_loss_real1 + recon_loss_real2),
            # "s_r": recon_loss_real1 * 100,
            # "s_f": recon_loss_fake1 * 100,
            # 'c_r': recon_loss_real2 * 100,
            # 'c_f': recon_loss_fake2 * 100,
            'kld': con_kld_con + con_kld_sti + sty_kld_con + sty_kld_sti,
            # 'delta': delta_loss * 0
        }

    def get_current_loss(self):
        return self.loss_stat

    def set_input(self, con, sti, sty):
        self.con = con.to(self.opt.device)
        self.sti = sti.to(self.opt.device)
        self.sty = sty.to(self.opt.device)

    # def get_sti(self, sti):
    #     mean_data = torch.mean(sti, dim=0)
    #     self.mean_data = mean_data.unsqueeze(0).expand(self.opt.batch_size, -1)
        
    def forward(self):
        self.out = self.model(self.con, self.sti, self.sty)

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_parameter(self):
        self.model.train()
        self.forward()
        self.compute_loss()
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

    def predict(self, pred_data, stim_data):
        self.model.eval()
        pred = self.adata2tensor(pred_data).to(self.opt.device)
        stim = self.adata2tensor(stim_data).to(self.opt.device)
        pred_out = self.model.predict(pred, stim)
        return self.tensor2numpy(pred_out)