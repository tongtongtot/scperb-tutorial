import torch
import random
import subprocess
import numpy as np
from tqdm import tqdm
from scipy import stats
from utils.utils import Utils
from options.option import options
from models.scgen_model import SCGEN
from dataloader.customDataset import customDataloader

def validation(opt, model_use = 'now'):
    model = SCGEN(opt)

    if model_use == 'best': 
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_best_epoch.pt')
    elif model_use == 'all' and opt.plot == True:
        validation(opt, model_use='best')
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
    else:
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')

    model.to(opt.device, non_blocking=True)

    pred_data, ctrl_data, stim_data, stim, cell_type = dataset.get_val_data()
    pred = model.predict(pred_data, ctrl_data, stim_data)

    if opt.plot == True:
        utils = Utils(opt)
        # ctrl_data, stim_data, cell_type = dataset.get_val_data()
        utils.plot_result(ctrl_data, stim_data, pred, cell_type)

    x = np.asarray(np.mean(stim, axis=0)).ravel()
    y = np.asarray(np.mean(pred, axis=0)).ravel()
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2 * 100.0

def train_model(opt, dataset):
    utils = Utils(opt)

    model = SCGEN(opt)
    
    if opt.resume == True:
        model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
    
    model.to(opt.device, non_blocking=True)
    
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = True, pin_memory = True)

    scores = 0
    best_model = 0
    pbar = tqdm(total=opt.epochs)

    for epoch in range(opt.epochs):
        loss = {}
        for idx, (x) in enumerate(dataloader):
            model.train()
            model.set_input(x)
            model.update_parameter()
            loss_dic = utils.get_loss(model)
            for (k, v) in loss_dic.items():
                if idx == 0:
                    loss[k] = v
                else:
                    loss[k] += v
            
            # print("pass_idx")

        model.save(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
        tmp_scores = validation(opt)
        best_model, scores = utils.bestmodel(scores, tmp_scores, epoch, best_model, model)
        
        utils.update_pbar(loss, scores, best_model, pbar, opt)

        # print("pass")


def fix_seed(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.seed)

if __name__ == '__main__':
    Opt = options()
    opt = Opt.init()
    utils = Utils(opt)
    fix_seed(opt)
    dataset = customDataloader(opt)
    
    if opt.download_data == True:
        command = "python3 DataDownloader.py"
        subprocess.call([command], shell=True)

    if opt.plot_only == True:
        opt.plot = True
        validation(opt)
    
    else:
        train_model(opt, dataset)
        opt.plot = True
        validation(opt)