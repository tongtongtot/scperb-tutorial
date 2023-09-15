import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.content_encoder = nn.Sequential(
            nn.Linear(opt.input_dim, opt.hidden_dim),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.ELU(),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.ELU(),
            nn.Linear(opt.hidden_dim, opt.context_latent_dim * 2)
        )
        self.style_encoder = nn.Sequential(
            nn.Linear(opt.input_dim, opt.hidden_dim),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.ELU(),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.ELU(),
            nn.Linear(opt.hidden_dim, opt.style_latent_dim * 2)
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(opt.input_dim, opt.context_latent_dim + opt.style_latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(opt.context_latent_dim + opt.style_latent_dim, opt.hidden_dim),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.ELU(),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.ELU(),
            nn.Linear(opt.hidden_dim, opt.input_dim),
            nn.ReLU()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def get_con_z(self, x):
        h = self.content_encoder(x)
        mu , logvar = torch.split(h, h.size(-1) // 2, dim = -1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def get_sty_z(self, x):
        h = self.style_encoder(x)
        mu , logvar = torch.split(h, h.size(-1) // 2, dim = -1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def predict(self, pred, stim_data):
        pred_con, mu1, var2 = self.get_con_z(pred)
        z_sti_sty, _, _ = self.get_sty_z(stim_data)
        # print(z_sti_sty.shape)
        z_sty_mean = z_sti_sty.mean(dim=0)
        _, z_dim = z_sti_sty.shape
        z_sty_batch = z_sty_mean.repeat((pred_con.shape[0],z_dim))

        sty = torch.ones(pred_con.shape[0], pred_con.shape[1] + z_dim)

        # print(z_sty_batch.shape)
        return self.decoder(torch.cat([pred_con, z_sty_batch], dim=1).to(self.opt.device))

    def forward(self, con, sti, sty):
        z_con_con, mu_con_con, var_con_con = self.get_con_z(con)
        z_con_sty, mu_con_sty, var_con_sty = self.get_sty_z(con)
        z_sti_con, mu_sti_con, var_sti_con = self.get_con_z(sti)
        z_sti_sty, mu_sti_sty, var_sti_sty = self.get_sty_z(sti)

        delta = self.delta_encoder(sty)

        real_con_recon = self.decoder(torch.cat([z_con_con, z_con_sty], dim=1))
        fake_con_recon = self.decoder(torch.cat([z_sti_con, z_con_sty], dim=1))
        real_sti_recon = self.decoder(torch.cat([z_sti_con, z_sti_sty], dim=1))
        fake_sti_recon = self.decoder(torch.cat([z_con_con, z_sti_sty], dim=1))

        real_delta = torch.cat([z_sti_con, z_sti_sty], dim=1) - torch.cat([z_con_con, z_con_sty], dim=1)

        return {
            "z_con_con": z_con_con, "mu_con_con": mu_con_con, "var_con_con": var_con_con,
            "z_con_sty": z_con_sty, "mu_con_sty": mu_con_sty, "var_con_sty": var_con_sty,
            "z_sti_con": z_sti_con, "mu_sti_con": mu_sti_con, "var_sti_con": var_sti_con,
            "z_sti_sty": z_sti_sty, "mu_sti_sty": mu_sti_sty, "var_sti_sty": var_sti_sty,
            "real_con_recon": real_con_recon, 
            "fake_con_recon": fake_con_recon,
            "real_sti_recon": real_sti_recon,
            "fake_sti_recon": fake_sti_recon,
            'delta': delta,
            'real_delta': real_delta
        }

