import torch
from torch import nn
from models.gimVi_vae import gimVi_vae
from torch.nn.functional import cosine_similarity

class gimVi_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = gimVi_vae(opt)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        self.mse_loss = torch.nn.MSELoss(reduction = 'sum')
        self.mse_loss2 = torch.nn.MSELoss(reduction = 'mean')
        self.loss_stat = {}
    
    def get_current_loss(self):
        return self.loss_stat

    def set_input(self, con, sti, sty):
        self.con = con.to(self.opt.device)
        self.sti = sti.to(self.opt.device)
        self.sty = sty.to(self.opt.device)

    def forward(self):
        self.model.train()
        self.out = self.model(self.con, self.sti, self.sty)

    def gram_matrix(self, feat):
        # print(feat.shape)
        b,d = feat.shape
        G = torch.mm(feat, feat.t()) # b * d * d * b
        return G.div(b * d)

    def get_reconstruction_loss(self, x, px):
        loss = ((x - px) ** 2).sum(dim=1)
        return loss

    def compute_loss(self):
        # compute cont_loss
        cont_loss1 = self.mse_loss2(self.out['st_cont1'], self.out['sc_cont1']) #+ 1 - cosine_similarity(self.out['st_cont1'], self.out['sc_cont1'], dim=1).mean()
        cont_loss2 = self.mse_loss2(self.out['st_cont2'], self.out['sc_cont2']) #+ 1 - cosine_similarity(self.out['st_cont2'], self.out['sc_cont2'], dim=1).mean()
        cont_loss = torch.mean(cont_loss1 + cont_loss2)

        # compute style_loss
        target_g1 = self.gram_matrix(self.out['st_style1']).detach()
        target_g2 = self.gram_matrix(self.out['st_style2']).detach()
        # print(self.out['fake_style1'].shape)
        style_loss1 = self.mse_loss(self.gram_matrix(self.out['fake_style1']), target_g1)
        style_loss2 = self.mse_loss(self.gram_matrix(self.out['fake_style2']), target_g2)
        style_loss = style_loss1 + style_loss2 

        # similarity loss between real and fake
        cs_loss1 = 1 - cosine_similarity(self.out['st_real'], self.out['st_fake'], dim=1).mean()
        cs_loss2 = 1 - cosine_similarity(self.out['st_real'], self.sti, dim=1).mean()
        # cs_loss2 = self.get_reconstruction_loss(self.out['st_real'], self.sti)
        # cs_loss1 = self.get_reconstruction_loss(self.out['st_fake'], self.con)
        # cs_test = self.mse_loss(self.out['st_real'], self.out['st_fake'])
        # cs_loss = torch.mean(cs_loss1 + cs_loss2 + cs_test * 0.5)
        cs_loss = torch.mean(cs_loss1 + cs_loss2)

        # self.loss = cont_loss + style_loss + cs_loss

        self.loss = cs_loss

        # print(self.loss)
        # print('con', cont_loss)
        # print('sty', style_loss)
        # print('cs', cs_loss)
        self.loss_stat = {
            'total_loss': self.loss.item(),
            'cont_loss': cont_loss.item(),
            'style_loss': style_loss.item(),
            'cs_loss': cs_loss.item()
        }

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_parameter(self):
        self.forward()
        self.compute_loss()
        self.backward()

    def tensor2numpy(self, data):
        return data.cpu().detach().numpy()

    def predict(self, con, sti):
        self.model.eval()
        self.out = self.model(con.to(self.opt.device), sti.to(self.opt.device), con.to(self.opt.device), istrain = False)
        return self.tensor2numpy(self.out['st_fake'])

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
