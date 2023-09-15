import os
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats
from utils.utils import Utils
from options.option import options
import random
from dataloader.originalDataset import originalDataset
from models.origin_model import original_model

def validation(opt, model_use = 'now'):

    opt.validation = True
    opt.model_use = model_use
    model = original_model(opt)

    if model_use == 'best': 
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_best_epoch.pt')
    elif model_use == 'all' and opt.plot == True:
        validation(opt, model_use='best')
        opt.validation = True
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
    else:
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')

    model.to(opt.device, non_blocking=True)

    pred, con, sti, stim_data, _ = dataset.get_val_data()

    pred = model.predict(pred, sti)
    
    # stim_data = dataset.get_real_stim()

    x = np.asarray(np.mean(stim_data, axis=0)).ravel()
    y = np.asarray(np.mean(pred, axis=0)).ravel()

    m, b, r_value, p_value, std_err = stats.linregress(x, y)

    if opt.plot == True:
        utils = Utils(opt)
        _, ctrl_data, stim_data, __, cell_type = dataset.get_val_data()
        utils.plot_result(ctrl_data, stim_data, pred, cell_type, model_use)

    opt.validation = False
    return r_value ** 2 * 100.0

def train_model(opt):
    # fix_seed(opt)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.seed)
    
    utils = Utils(opt)

    model = original_model(opt)
    
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
            model.update_parameter()
            loss_dic = utils.get_loss(model)
            for (k, v) in loss_dic.items():
                if idx == 0:
                    loss[k] = v
                else:
                    loss[k] += v

        model.save(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
        tmp_scores = validation(opt)
        best_model, scores = utils.bestmodel(scores, tmp_scores, epoch, best_model, model)
        
        utils.update_pbar(loss, scores, best_model, pbar, opt)

def fix_seed(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.seed)

if __name__ == '__main__':
    Opt = options()
    opt = Opt.init()
    dataset = originalDataset(opt)
    fix_seed(opt)

    if opt.plot_only == True:
        opt.plot = True
        validation(opt, 'all')
    else:
        train_model(opt)
        opt.plot = True
        validation(opt, 'all')