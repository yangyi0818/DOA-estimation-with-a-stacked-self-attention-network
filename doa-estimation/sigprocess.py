import torch as th
import torch.nn as nn
import numpy as np

EPSILON = 1e-8

class STFT(nn.Module):
    def __init__(self, fftsize, window_size, stride, trainable=False):
        super(STFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride
        self.window_func = np.hanning(self.window_size)

        fcoef_r = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        fcoef_i = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        for w in range(self.fftsize//2+1):
            for t in range(self.window_size):
                fcoef_r[w, 0, t] = np.cos(2. * np.pi * w * t / self.fftsize)
                fcoef_i[w, 0, t] = -np.sin(2. * np.pi * w * t / self.fftsize)

        fcoef_r = fcoef_r * self.window_func
        fcoef_i = fcoef_i * self.window_func
        self.fcoef_r = th.tensor(fcoef_r, dtype=th.float)
        self.fcoef_i = th.tensor(fcoef_i, dtype=th.float)
        self.encoder_r = nn.Conv1d(1, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)
        self.encoder_i = nn.Conv1d(1, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)
        self.encoder_r.weight = th.nn.Parameter(self.fcoef_r)
        self.encoder_i.weight = th.nn.Parameter(self.fcoef_i)

        if trainable:
            self.encoder_r.weight.requires_grad = True
            self.encoder_i.weight.requires_grad = True
        else:
            self.encoder_r.weight.requires_grad = False
            self.encoder_i.weight.requires_grad = False

    def forward(self, input): # (B, 1, n_sample)

        spec_r = self.encoder_r(input)
        spec_i = self.encoder_i(input)
        output = th.stack([spec_r,spec_i],dim=-1)
        output = output.permute([0, 2, 1, 3])

        return output # (B,T,F,2)

class ISTFT(nn.Module):
    def __init__(self, fftsize, window_size, stride, trainable=False):
        super(ISTFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride

        gain_ifft = (2.0*self.stride) / self.window_size
        self.window_func = gain_ifft * np.hanning(self.window_size)

        coef_cos = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        coef_sin = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        for w in range(self.fftsize//2+1):
            alpha = 1.0 if w==0 or w==fftsize//2 else 2.0
            alpha /= fftsize
            for t in range(self.window_size):
                coef_cos[w, 0, t] = alpha * np.cos(2. * np.pi * w * t / self.fftsize)
                coef_sin[w, 0, t] = alpha * np.sin(2. * np.pi * w * t / self.fftsize)

        self.coef_cos = th.tensor(coef_cos * self.window_func, dtype=th.float)
        self.coef_sin = th.tensor(coef_sin * self.window_func, dtype=th.float)
        self.decoder_re = nn.ConvTranspose1d(self.fftsize//2+1, 1, self.window_size, bias=False, stride=self.stride)
        self.decoder_im = nn.ConvTranspose1d(self.fftsize//2+1, 1, self.window_size, bias=False, stride=self.stride)
        self.decoder_re.weight = th.nn.Parameter(self.coef_cos)
        self.decoder_im.weight = th.nn.Parameter(self.coef_sin)

        if trainable:
            self.decoder_re.weight.requires_grad = True
            self.decoder_im.weight.requires_grad = True
        else:
            self.decoder_re.weight.requires_grad = False
            self.decoder_im.weight.requires_grad = False

    def forward(self, input): # (B,T,F,2)
        input = input.permute([0, 2, 1, 3]) # (B,F,T,2)
        real_part = input[:,:,:,0]
        imag_part = input[:,:,:,1]

        time_cos = self.decoder_re(real_part)
        time_sin = self.decoder_im(imag_part)
        output = time_cos - time_sin

        return output  # (B, 1, n_sample)
