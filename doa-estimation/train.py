import os
import argparse
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import Librispeech_Dataset
from asteroid.engine.optimizers import make_optimizer
from system import System, label_loss
from model import make_model_and_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--train_dirs", type=str, required=True, help="Training dataset")
parser.add_argument("--val_dirs", type=str, required=True, help="Val dataset")
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)

def main(conf):
    train_dir = conf["main_args"]['train_dirs']
    val_dir = conf["main_args"]['val_dirs']

    rirNO_train = len(os.listdir(train_dir))
    rirNO_val = len(os.listdir(val_dir))
    
    train_set = Librispeech_Dataset(
        train_dir,
        rirNO_train,
        trainingNO = 10000,
        segment=8,
        channel=[0,1,2,3],
    )

    val_set = Librispeech_Dataset(
        val_dir,
        rirNO_val,
        trainingNO = 2000,
        segment=8,
        channel=[0,1,2,3],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        worker_init_fn=_worker_init_fn_
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        worker_init_fn=_worker_init_fn_
    )
    
    # Define model and optimizer
    model, optimizer = make_model_and_optimizer(conf)

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = label_loss()

    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )
    
    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    early_stopping = False
    if conf["training"]["early_stop"]:
        early_stopping = EarlyStopping(monitor="val_loss", patience=30, verbose=True)

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        checkpoint_callback=checkpoint,
        #resume_from_checkpoint='exp/epoch=46.ckpt',
        early_stop_callback=early_stopping,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend="dp",
        train_percent_check=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)
        
if __name__ == "__main__":

    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)
