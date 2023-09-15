import torch
from torch import nn

class SCGENVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        input_dim = opt.input_dim
        hidden_dim = opt.hidden_dim
        latent_dim = opt.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
        )
        
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.logvar_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(opt.drop_out),
            nn.Linear(hidden_dim, input_dim),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.img_size = input_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def get_z(self, x):
        hidden = self.encoder(x)
        mu = self.mu_encoder(hidden)
        logvar = self.logvar_encoder(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward_sig(self, x):
        z, mu, logvar = self.get_z(x)
        outs = {
            'z': [z, mu, logvar],
            'output': self.decoder(z)
        }
        return outs

    def forward_dou(self, con, sti):
        z_con, mu_con, logvar_con = self.get_z(con)
        z_sti, mu_sti, logvar_sti = self.get_z(sti)
        outs = {
            'z_con': [z_con, mu_con, logvar_con],
            'z_sti': [z_sti, mu_sti, logvar_sti],
            'output': [self.decoder(z_con), self.decoder(z_sti)],
        }
        return outs

    def forward(self, x):
        if self.opt.model_use == 'vae_test1':
            return self.forward_dou(x[0].to(self.opt.device), x[1].to(self.opt.device))
        else:
            return self.forward_sig(x.to(self.opt.device))