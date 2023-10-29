import torch
from torch import nn

class mlp_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.l = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, use_norm=True):
        x = self.l(x)
        if use_norm:
            x = self.norm(x)
            x = self.relu(x)
        return x

class gimVi_vae(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # self.st_enc1_cont = mlp_simple(opt.input_dim, opt.hidden_dim)
        # self.st_enc2_cont = mlp_simple(opt.hidden_dim, opt.hidden_dim)

        # self.st_enc1_style = mlp_simple(opt.input_dim, opt.hidden_dim)
        # self.st_enc2_style = mlp_simple(opt.hidden_dim, opt.hidden_dim)

        # self.st_dec2 = mlp_simple(opt.hidden_dim, opt.hidden_dim)
        # self.st_dec1 = mlp_simple(opt.hidden_dim, opt.input_dim)

        # self.sc_enc2_cont = mlp_simple(opt.input_dim, opt.hidden_dim)
        # self.sc_enc1_cont = mlp_simple(opt.hidden_dim, opt.hidden_dim)

        # self.enc_style2 = mlp_simple(opt.input_dim, opt.hidden_dim)
        # self.enc_style1 = mlp_simple(opt.input_dim, opt.hidden_dim)

        self.opt = opt

        self.st_enc1_cont = mlp_simple(opt.input_dim, 256)
        self.st_enc2_cont = mlp_simple(256, 10)

        self.st_enc1_style = mlp_simple(opt.input_dim, 256)
        self.st_enc2_style = mlp_simple(256, 10)

        self.st_dec2 = mlp_simple(10, 256)
        self.st_dec1 = mlp_simple(256, opt.input_dim)

        self.sc_enc1_cont = mlp_simple(opt.input_dim, 256)
        self.sc_enc2_cont = mlp_simple(256, 10)

        self.enc_style1 = mlp_simple(opt.input_dim, 256)
        self.enc_style2 = mlp_simple(256, 10)
    
    def forward(self, con, sti, sty, istrain = True):
        # ststyle = torch.ones(self.opt.input_dim).to(self.opt.device)
        if istrain:
            # generate st cont
            st_cont1 = self.st_enc1_cont(sti)
            st_cont2 = self.st_enc2_cont(st_cont1)

            # generate st style
            st_style1 = self.st_enc1_style(sti)
            st_style2 = self.st_enc2_style(st_style1)

            # generate sc cont
            sc_cont1 = self.sc_enc1_cont(con)
            sc_cont2 = self.sc_enc2_cont(sc_cont1)

            # generate fake style
            # fake_style1 = self.enc_style1(sty)
            # fake_style2 = self.enc_style2(fake_style1)
            fake_style1 = self.enc_style1(con)
            fake_style2 = self.enc_style2(fake_style1)

            # real 
            real_st_up2 = self.st_dec2(st_cont2 * st_style2)
            real_st_up1 = self.st_dec1(real_st_up2 + st_cont1 * st_style1, use_norm=False)

            # fake
            fake_st_up2 = self.st_dec2(sc_cont2 * (st_style2 - fake_style2).mean())
            fake_st_up1 = self.st_dec1(fake_st_up2 + sc_cont1 * (st_style1 - fake_style1).mean() , use_norm=False)

            return {
                'st_cont1': st_cont1, 'st_cont2': st_cont2,
                'sc_cont1': sc_cont1, 'sc_cont2': sc_cont2,
                'st_style1': st_style1, 'st_style2': st_style2,
                'fake_style1': fake_style1, 'fake_style2': fake_style2,
                'st_real': real_st_up1, 'st_fake': fake_st_up1
            }

        else:
            # only have sc and ststyle
            # generate sc_cont
            sc_cont1 = self.sc_enc1_cont(con)
            sc_cont2 = self.sc_enc2_cont(sc_cont1)

            fake_style1 = self.enc_style1(sti)
            fake_style2 = self.enc_style2(fake_style1)

            fake_st_up2 = self.st_dec2(sc_cont2 * fake_style2)
            fake_st_up1 = self.st_dec1(fake_st_up2 + sc_cont1 * fake_style1, use_norm=False)
            return {
                'st_fake': fake_st_up1
            }
