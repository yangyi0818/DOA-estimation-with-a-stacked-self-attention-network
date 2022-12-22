#!/user/bin/env python
# yangyi@2020-2022
# real-time doa estimation via self-attention

import torch
import torch.nn as nn
import numpy as np

from sigprocess import STFT, ISTFT
from base import dense_block, attention_block
from asteroid.engine.optimizers import make_optimizer

def make_model_and_optimizer(conf):
    model = proposed()
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer


class proposed(nn.Module):
    def __init__(self, fftsize=512, window_size=400, stride=100, channel=4, causal=True):
        super(proposed, self).__init__()
        bins = fftsize // 2
        self.channel = channel
        self.causal = causal

        self.stft = STFT(fftsize=fftsize, window_size=window_size, stride=stride, trainable=False)
        self.input_conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(8,8,[1,1],[1,1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8)
            )

        # dense conv block
        self.conv_block = nn.ModuleList()
        for i in range (4):
            self.conv_block.append(dense_block(in_channels=8, out_channels=8, kernel_size=[2,3], stride=[1,1], padding=[1,1]))
            
        # self-attention block
        self.shared_block = nn.ModuleList()
        for i in range (4):
            self.shared_block.append(attention_block(in_channels=bins//(4**i), out_channels=bins//(4**(i+1))))
            self.shared_block.append(attention_block(in_channels=8, out_channels=8))
            self.shared_block.append(attention_block(in_channels=bins//(4**(i+1)), out_channels=bins//(4**(i+1))))

        self.re_fc_layer = nn.Linear(8,6)


    def forward(self, x):    # b n c

        x = x.transpose(1,2) # b c n
        xs = self.stft(x[:,[0],:])[...,1:,:].unsqueeze(1) # b 1 t f 2
        for i in range(1,self.channel):
            xs = torch.cat((xs,self.stft(x[:,[i],:])[...,1:,:].unsqueeze(1)),1) # b c t f 2
        feat_in = torch.cat((xs[...,0], xs[...,1]), 1)                          # b 2c t f

        # step1:change channel dim
        x_in = self.input_conv_layer(feat_in) # b 2c t f

        for i in range (4):
            # step2:dense block
            x_out1 = self.conv_block[i](x_in) # (B,8,T,256) (B,8,T,64) (B,8,T,16) (B,8,T,4)

            # step3:self-attention T
            x_in2 = x_out1.permute(0,3,1,2)   # (B,256,8,T)
            x_out2, plot_weight_t = self.shared_block[0+i*3](x_in2, causal=self.causal) # (B,64,8,T)
            x_out2 = x_out2.permute(0,2,3,1)
            
            # step4:self-attention F
            x_in3 = x_out2                    # (B,8,T,64)
            x_out3, plot_weight_f = self.shared_block[1+i*3](x_in3, causal=self.causal) # (B,8,T,64)

            x_in = x_out3

        x_out = x_in.squeeze(-1).transpose(1,2) # b t 8
        y = self.re_fc_layer(x_out)             # b t 6

        return y.mean(1)


if __name__ == "__main__":
    import torch
    from thop import profile
    from thop import clever_format

    model = proposed()
    x = torch.randn(1, 16000*8, 7) # b n c
    macs, params = profile(model, inputs=(x))
    macs, params = clever_format([macs, params], "%.3f")

    print('macs:', macs)
    print('params:', params)
