import torch
import torch.nn as nn
from numpy import inf

"""
# ref: 2021, Attention is All You Need in Speech Separation
#      https://arxiv.org/abs/2010.13154v2
# ref: 2021, Dense CNN with Self-Attention for Time-Domain Speech Enhancement
#      https://arxiv.org/abs/2009.01941
# time delay:stft 25ms
# positional encoding may be added into it. original chunk_size=250
"""
class dense_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=[2,3],stride=[1,1],padding=[1,1],bias=False):
        super(dense_block,self).__init__()
        self.shared_block = nn.ModuleList()
        for i in range(5):
            self.shared_block.append(nn.Conv2d(in_channels*(i+1),out_channels,kernel_size,stride,padding,bias=bias))
            self.shared_block.append(nn.PReLU())
            self.shared_block.append(torch.nn.BatchNorm2d(out_channels))

    def forward(self,x):
        for i in range(5):
            x1 = self.shared_block[0+3*i](x)[:,:,:x.size()[-2],:]
            x2 = self.shared_block[1+3*i](x1)
            x3 = self.shared_block[2+3*i](x2)
            x = torch.cat((x,x3),1)

        return x3

class attention_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=[1,1],stride=[1,1],padding=[0,0]):
        super(attention_block,self).__init__()
        self.conv_Q = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.conv_K = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.conv_V = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

    def forward(self,x,causal=False):
        Q = self.conv_Q(x).reshape(x.size()[0],-1,x.size()[3])
        K = self.conv_K(x).reshape(x.size()[0],-1,x.size()[3])
        V = self.conv_V(x).reshape(x.size()[0],-1,x.size()[3])
        # scaled dpd
        _attention_weight = (torch.einsum('ikj,ijl->ikl', [Q.permute(0,2,1), K]) / (Q.size()[1])**0.5)

        if (causal==True):
            # causal attention
            _attention_weight = torch.triu(_attention_weight)
            mask = torch.ones_like(_attention_weight) * float(-inf)
            _attention_weight = torch.where(_attention_weight != 0, _attention_weight, mask)

        attention_weight = _attention_weight.softmax(dim=-1)
        x_out = torch.einsum('ijl,ilk->ijk', [V, attention_weight.permute(0,2,1)])
        x_out = x_out.reshape(x.size()[0],-1,x.size()[2],x.size()[3])
        plot_weight = _attention_weight

        return x_out, plot_weight
      
"""
dataset
"""
import numpy as np
import math
EPS=1e-8

def rms(y):
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor
  
def get_label(_source_positions, _sensor_positions, usage):
    """
    Extract Label of Raw Wav.
    Arguments:
        _source_positions: source positions, 3 x 2 x channel
        _sensor_positions: sensor positions, 3 x 2 x channel
        loss_type: categorical or cartesian
    Return:
        y: (1,)    if categorical
           (3,2)   if cartesian
    """
    ''' 
    _source_positions = np.array([[[1.0], [1.0]],
                                  [[0.0], [1.0]],
                                  [[0.0], [1.0]]])
    _sensor_positions = np.array([[[0.0,  1.0, -1.0,  0.0]],
                                  [[0.0,  1.0,  0.0,  0.0]],
                                  [[0.0,  0.0,  0.0,  0.0]]])
    '''
    #print('src:',_source_positions)
    #print('sen:',_sensor_positions)
    if (usage == 'simu'):
        # step1:translation
        x0 = _source_positions[0,:,0]; y0 = _source_positions[1,:,0]; z0 = _source_positions[2,:,0]
        x1 = -(_sensor_positions[0,:,0] - x0); y1 = -(_sensor_positions[1,:,0] - y0); z1 = -(_sensor_positions[2,:,0] - z0)

        ref_x0 = _sensor_positions[0,:,1]; ref_y0 = _sensor_positions[1,:,1]; ref_z0 = _sensor_positions[2,:,1]
        ref_x1 = -(_sensor_positions[0,:,0] - ref_x0); ref_y1 = -(_sensor_positions[1,:,0] - ref_y0); ref_z1 = -(_sensor_positions[2,:,0] - ref_z0)

        # step2:rotation-azimuth
        theta = np.arctan2(ref_y1,ref_x1)
        x2 = x1 * np.cos(theta) + y1 * np.sin(theta)
        y2 = y1 * np.cos(theta) - x1 * np.sin(theta)
        z2 = z1

        # step3:rotation-elevation
        phi = np.arctan2(ref_z1,np.sqrt(ref_x1 ** 2 + ref_y1 ** 2))
        x3 = x2 * np.cos(phi) + z2 * np.sin(phi)
        y3 = y2
        z3 = z2 * np.cos(phi) - x2 * np.sin(phi)

    elif (usage == 'dummy'):
        x0 = (_sensor_positions[0,:,0] + _sensor_positions[0,:,2]) / 2; y0 = _sensor_positions[1,:,0]
        y3 = _source_positions[1,:,0] - y0
        x3 = _source_positions[0,:,0] - x0
        z3 = _source_positions[2,:,0] - z0

    y = np.vstack((x3,y3,z3)) # 3 src
    #print('dis:',dis)

    return y
