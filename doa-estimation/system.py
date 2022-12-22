import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from argparse import Namespace
from typing import Callable, Optional
from torch.optim.optimizer import Optimizer
from asteroid.utils import flatten_dict

from torch.nn.modules.loss import _Loss
from asteroid.utils.deprecation_utils import DeprecationMixin
EPS = 1e-8

class System(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        self.hparams = Namespace(**self.config_to_hparams(config))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        inputs, mix_name, label, mix_way = batch
        est_label = self(inputs)
        loss, loss_dict = self.loss_func(label, est_label, mix_way)
        return loss, loss_dict

    def training_step(self, batch, batch_nb):
        loss, loss_dict = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = loss_dict
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, loss_dict = self.common_step(batch, batch_nb, train=False)
        tensorboard_logs = loss_dict
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def optimizer_step(self, *args, **kwargs) -> None:
        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            for sched in self.scheduler:
                if isinstance(sched, dict) and sched["interval"] == "batch":
                    sched["scheduler"].step()  # call step on each batch scheduler
            super().optimizer_step(*args, **kwargs)
            
    def configure_optimizers(self):
        """ Required by pytorch-lightning. """

        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            epoch_schedulers = []
            for sched in self.scheduler:
                if not isinstance(sched, dict):
                    epoch_schedulers.append(sched)
                else:
                    assert sched["interval"] in [
                        "batch",
                        "epoch",
                    ], "Scheduler interval should be either batch or epoch"
                    if sched["interval"] == "epoch":
                        epoch_schedulers.append(sched)
            return [self.optimizer], epoch_schedulers
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
      
    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic


class label_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, label, est_label, mix_way):
        label_loss = 0

        # permutation
        for batch in range(label.size(0)):
            if mix_way[batch] == 'single':
                label_loss += self.mse_loss(est_label[:,:3], label[:,:,0])
            else:
                label_loss1 = self.mse_loss(est_label[[batch],:3], label[[batch],:,0]) + self.mse_loss(est_label[[batch],3:], label[[batch],:,1])
                label_loss2 = self.mse_loss(est_label[[batch],3:], label[[batch],:,0]) + self.mse_loss(est_label[[batch],:3], label[[batch],:,1])
                if label_loss1 < label_loss2:
                    label_loss += label_loss1
                else:
                    label_loss += label_loss2
        label_loss /= label.size(0)


        # accuracy
        MAE1, MAE2 = 0, 0
        label_azimuth1 = torch.atan2(label[:,1,0], label[:,0,0]) / np.pi * 180
        label_azimuth2 = torch.atan2(label[:,1,1], label[:,0,1]) / np.pi * 180
        est_azimuth1 = torch.atan2(est_label[:,1], est_label[:,0]) / np.pi *180
        est_azimuth2 = torch.atan2(est_label[:,4], est_label[:,3]) / np.pi *180

        error_azimuth11 = torch.abs(label_azimuth1 - est_azimuth1)
        error_azimuth22 = torch.abs(label_azimuth2 - est_azimuth2)
        error_azimuth12 = torch.abs(label_azimuth1 - est_azimuth2)
        error_azimuth21 = torch.abs(label_azimuth2 - est_azimuth1)

        for batch in range (label.size(0)):
            if (error_azimuth11[batch] > 180):
                error_azimuth11[batch] = 360 - error_azimuth11[batch]
            if (error_azimuth22[batch] > 180):
                error_azimuth22[batch] = 360 - error_azimuth22[batch]
            if (error_azimuth12[batch] > 180):
                error_azimuth12[batch] = 360 - error_azimuth12[batch]
            if (error_azimuth21[batch] > 180):
                error_azimuth21[batch] = 360 - error_azimuth21[batch]

        for batch in range (label.size(0)):
            if error_azimuth11[batch]+error_azimuth22[batch] < error_azimuth12[batch]+error_azimuth21[batch]:
                MAE1 += error_azimuth11[batch]
                MAE2 += error_azimuth22[batch]
            else:
                MAE1 += error_azimuth12[batch]
                MAE2 += error_azimuth21[batch]

        MAE1 = MAE1 / label.size(0)
        MAE2 = MAE2 / label.size(0)

        loss_dict = dict(sig_loss=label_loss.mean(), MAE1=MAE1, MAE2=MAE2)

        return label_loss.mean(), loss_dict
