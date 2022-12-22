import os
import random
import soundfile as sf
import torch
from torch import nn
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm

from asteroid import torch_utils
from asteroid.utils import tensors_to_device
from model import make_model_and_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, required=True, help="Test directory including the json files")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")

def load_best_model(model, exp_dir):
    try:
        with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt]
        all_ckpt.sort(key=lambda x:x[1])
        best_model_path = os.path.join(exp_dir, 'checkpoints', all_ckpt[-1][0])
    print( 'LOADING from ',best_model_path)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location='cpu')
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], model)
    model = model.eval()
    return model
  
  
def main(conf):
    azimuth_resolution = np.array([2.5,5,10,15,20,25,30,35,40])
    True_est_azimuth1, True_est_azimuth2, True_est_azimuth = np.zeros(9), np.zeros(9), np.zeros(9)
    azimuth_mean_loss1, azimuth_mean_loss2, azimuth_mean_loss = 0, 0, 0

    model, _ = make_model_and_optimizer(train_conf)
    model = load_best_model(model, conf['exp_dir'])
    testset = conf['test_dir']

    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    dlist = os.listdir(testset)
    pbar = tqdm(range(len(dlist)))
    torch.no_grad().__enter__()
    for idx in pbar:
        test_wav = np.load(testset + dlist[idx])
        mix, label, mix_way = tensors_to_device([torch.from_numpy(test_wav['mix']), torch.from_numpy(test_wav['label']), test_wav['mix_way']], device=model_device)
        est_label = model(mix[None])

        # unbiased
        if mix_way == 'single':

        # biased
        #if mix_way == 'single' or mix_way == 'dominant':
            label = label[:,0]; est_label = est_label[:,:3]

            # accuracy
            label_azimuth = (torch.atan2(label[1], label[0]) / np.pi * 180).cpu().numpy()
            est_azimuth = (torch.atan2(est_label[0,1], est_label[0,0]) / np.pi *180).cpu().numpy()

            error_azimuth = np.abs(label_azimuth - est_azimuth)
            if (error_azimuth > 180):
                error_azimuth = 360 - error_azimuth
            True_est_azimuth += (error_azimuth <= azimuth_resolution)
            azimuth_mean_loss += error_azimuth
            
            pbar.set_description(" {} {} {} {}".format('%.1f'%(azimuth_mean_loss / (idx+1)), '%.1f'%(error_azimuth), '%.1f'%(label_azimuth), '%.1f'%(est_azimuth)))

        else:
            label1 = label[:,0]; est_label1 = est_label[:,:3]
            label2 = label[:,1]; est_label2 = est_label[:,3:]

            # accuracy
            label_azimuth1 = (torch.atan2(label1[1], label1[0]) / np.pi * 180).cpu().numpy()
            label_azimuth2 = (torch.atan2(label2[1], label2[0]) / np.pi * 180).cpu().numpy()
            est_azimuth1 = (torch.atan2(est_label1[0,1], est_label1[0,0]) / np.pi *180).cpu().numpy()
            est_azimuth2 = (torch.atan2(est_label2[0,1], est_label2[0,0]) / np.pi *180).cpu().numpy()

            error_azimuth11 = np.abs(label_azimuth1 - est_azimuth1)
            error_azimuth22 = np.abs(label_azimuth2 - est_azimuth2)
            error_azimuth12 = np.abs(label_azimuth1 - est_azimuth2)
            error_azimuth21 = np.abs(label_azimuth2 - est_azimuth1)
            if (error_azimuth11 > 180):
                error_azimuth11 = 360 - error_azimuth11
            if (error_azimuth22 > 180):
                error_azimuth22 = 360 - error_azimuth22
            if (error_azimuth12 > 180):
                error_azimuth12 = 360 - error_azimuth12
            if (error_azimuth21 > 180):
                error_azimuth21 = 360 - error_azimuth21

            if error_azimuth11+error_azimuth22 < error_azimuth12+error_azimuth21:
                True_est_azimuth1 += (error_azimuth11 <= azimuth_resolution)
                azimuth_mean_loss1 += error_azimuth11
                True_est_azimuth2 += (error_azimuth22 <= azimuth_resolution)
                azimuth_mean_loss2 += error_azimuth22
                error_azimuth = error_azimuth11 + error_azimuth22
            else:
                True_est_azimuth1 += (error_azimuth12 <= azimuth_resolution)
                azimuth_mean_loss1 += error_azimuth12
                True_est_azimuth2 += (error_azimuth21 <= azimuth_resolution)
                azimuth_mean_loss2 += error_azimuth21
                error_azimuth = error_azimuth12 + error_azimuth21

            azimuth_mean_loss = (azimuth_mean_loss1 + azimuth_mean_loss2) / 2
            error_azimuth /= 2
            True_est_azimuth = (True_est_azimuth1 + True_est_azimuth2) / 2

            pbar.set_description(" {} {} {} {} {} {}".format('%.1f'%(azimuth_mean_loss / (idx+1)), '%.1f'%(error_azimuth), \
                                                             '%.1f'%(label_azimuth1), '%.1f'%(label_azimuth2), '%.1f'%(est_azimuth1), '%.1f'%(est_azimuth2)))

    azimuth_mean_loss /= len(dlist)
    print('azimuth MAE in degree: ', '%.2f'%(azimuth_mean_loss))
    for i in range (len(azimuth_resolution)):
        print('Acc. on azimuth resolution ', azimuth_resolution[i], ' : ', '%.3f'%(True_est_azimuth[i]/len(dlist)))
        
        
if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
