import torch
from torch.utils import data
import numpy as np
import os
import soundfile as sf
import math
import random
import shutil

from base import rms, get_amplitude_scaling_factor, get_label
from sms_wsj.database.create_rirs import config, scenarios, rirs
from sms_wsj.reverb.reverb_utils import convolve

EPS=1e-8

class Librispeech_Dataset(data.Dataset):
    def __init__(
        self,
        reverb_matrixs_dir,
        rirNO = 5,
        trainingNO = 5000,
        segment = 6,
        channel = [0,1,2,3],
        overlap = [0.0, 0.1, 0.2, 0.3, 0.4],
        raw_dir = '/path/to/LibriSpeech/filelist-all/',
        noise_wav = '/path/to/noise/',
    ):
        super(Librispeech_Dataset, self).__init__()
        self.reverb_matrixs_dir = reverb_matrixs_dir
        self.rirNO = rirNO
        self.trainingNO = trainingNO
        self.segment = segment
        self.channel = channel
        self.overlap = overlap
        self.raw_dir = raw_dir
        self.noise_wav = noise_wav

    def __len__(self):
        return self.trainingNO

    def add_reverb(self,raw_dir1,raw_dir2,h_use):
        with open(raw_dir1,'r') as fin1:
            with open(raw_dir2,'r') as fin2:
                wav1 = fin1.readlines()
                wav2 = fin2.readlines()
                choose_wav = True
                while(choose_wav):
                    i = np.random.randint(0,len(wav1))
                    j = np.random.randint(0,len(wav2))
                    w1,fs = sf.read(os.path.join('/path/to/LibriSpeech', wav1[i].rstrip("\n")), dtype="float32")
                    w2,fs = sf.read(os.path.join('/path/to/LibriSpeech', wav2[j].rstrip("\n")), dtype="float32")
                    seg_len = int(fs * self.segment)
                    if (w1.shape[0] > seg_len + 1 and w2.shape[0] > seg_len + 1):
                        choose_wav = False

                w1_con = convolve(w1, h_use[0,:,:]).T
                w2_con = convolve(w2, h_use[1,:,:]).T
                
                SIR = random.uniform(-5,5)
                scalar=get_amplitude_scaling_factor(w1_con, w2_con, snr = SIR)
                w2_con = w2_con / scalar

                mix_way = np.random.choice(['single','dominant'], size=1, replace=False)
                mix_name = mix_way[0] + '-' + os.path.basename(raw_dir1)[:-4] + '-' + os.path.basename(raw_dir2)[:-4] + '.wav'
                if (mix_way == 'single'):
                    rand_start1 = np.random.randint(0, w1.shape[0] - seg_len)
                    stop1 = int(rand_start1 + seg_len)

                    mix_reverb = w1_con[rand_start1:stop1,:]
                    s1_reverb = w1_con[rand_start1:stop1,:]
                    s2_reverb = np.zeros_like(w1_con[rand_start1:stop1,:])

                if (mix_way == 'partial'):
                    rand_start1 = np.random.randint(0, w1.shape[0] - seg_len*0.75)
                    rand_start2 = np.random.randint(0, w2.shape[0] - seg_len*0.75)
                    stop1 = int(rand_start1 + seg_len*0.75)
                    stop2 = int(rand_start2 + seg_len*0.75)

                    mix_reverb = np.concatenate([w1_con[rand_start1:rand_start1 + int(seg_len*0.25),:], \
                                                 w1_con[rand_start1 + int(seg_len*0.25):stop1,:] + w2_con[rand_start2:rand_start2 + int(seg_len*0.5),:], \
                                                 w2_con[rand_start2 + int(seg_len*0.5):stop2,:]], axis=0)
                    s1_reverb = np.concatenate([w1_con[rand_start1:stop1,:],np.zeros_like(w2_con[rand_start2 + int(seg_len*0.5):stop2,:])], axis=0)
                    s2_reverb = np.concatenate([np.zeros_like(w1_con[rand_start1:rand_start1 + int(seg_len*0.25),:]),w2_con[rand_start2:stop2,:]], axis=0)

                if (mix_way == 'dominant'):
                    rand_start1 = np.random.randint(0, w1.shape[0] - seg_len)
                    rand_start2 = np.random.randint(0, w2.shape[0] - seg_len*0.5)
                    stop1 = int(rand_start1 + seg_len)
                    stop2 = int(rand_start2 + seg_len*0.5)

                    mix_reverb = np.concatenate([w1_con[rand_start1:rand_start1 + int(seg_len*0.5),:], \
                                                 w1_con[rand_start1 + int(seg_len*0.5):stop1,:] + w2_con[rand_start2:stop2,:]], axis=0)
                    s1_reverb = w1_con[rand_start1:stop1,:]
                    s2_reverb = np.concatenate([np.zeros_like(w1_con[rand_start1:rand_start1 + int(seg_len*0.5),:]),w2_con[rand_start2:stop2,:]], axis=0)
                    
                if (mix_way == 'sequential'):
                    rand_start1 = np.random.randint(0, w1.shape[0] - seg_len*0.5)
                    rand_start2 = np.random.randint(0, w2.shape[0] - seg_len*0.5)
                    stop1 = int(rand_start1 + seg_len*0.5)
                    stop2 = int(rand_start2 + seg_len*0.5)

                    mix_reverb = np.concatenate([w1_con[rand_start1:stop1,:],w2_con[rand_start2:stop2,:]], axis=0)
                    s1_reverb = np.concatenate([w1_con[rand_start1:stop1,:],np.zeros_like(w2_con[rand_start2:stop2,:])], axis=0)
                    s2_reverb = np.concatenate([np.zeros_like(w1_con[rand_start1:stop1,:]),w2_con[rand_start2:stop2,:]], axis=0)

        return mix_reverb, s1_reverb, s2_reverb, mix_name, mix_way
      
    def add_noise(self,mix_reverb,w_n):
        # dynamic SNR
        SNR = random.uniform(5,25)       
        x = []
        for item in mix_reverb:
            x.append(item[0])
        rand_start = np.random.randint(0,len(w_n)-len(x))
        stop = rand_start + len(x)
        scalar = get_amplitude_scaling_factor(x, w_n, snr = SNR)

        mix_noise = mix_reverb + (w_n[rand_start:stop]/scalar)[None].transpose()

        return mix_noise
      
    def __getitem__(self,idx):
        raw_list = os.listdir(self.raw_dir)
        SpeakerNo = len(raw_list)

        speaker1 = np.random.randint(0,SpeakerNo)
        speaker2 = np.random.randint(0,SpeakerNo)
        while (speaker1 == speaker2):
            speaker2 = np.random.randint(0,SpeakerNo)
        raw_dir1 = self.raw_dir+raw_list[speaker1]
        raw_dir2 = self.raw_dir+raw_list[speaker2]

        choose_rir = np.random.randint(0,self.rirNO)
        rand_rir = np.load(self.reverb_matrixs_dir + str(choose_rir).zfill(4) + '.npz')
        h_use, _source_positions, _sensor_positions, = rand_rir['h'], rand_rir['source_positions'], rand_rir['sensor_positions']

        # step1:add reverb to utterance 
        mix_reverb, s1_reverb, s2_reverb, mix_name, mix_way = self.add_reverb(raw_dir1,raw_dir2,h_use[:,self.channel,:])

        # step2:add noise
        w_n, _ = sf.read(self.noise_wav, dtype="float32")
        mix_noise = self.add_noise(mix_reverb,w_n)
        mix_noise = mix_noise.transpose()

        mixture = torch.from_numpy(np.array(mix_noise).astype(np.float32)).permute(1,0)

        _source_positions = _source_positions[...,None]                                # (3,src,channel)
        _sensor_positions = _sensor_positions[:,self.channel][:,None]                  # (3,src,channel)
        # 3d-to-2d 2021.09.21
        _source_positions[2] = _sensor_positions[2,0,0]

        label = get_label(_source_positions, _sensor_positions, usage='simu')        # (3,src) 
        label = torch.from_numpy(label.astype(np.float32))
        
        if (mix_way[0]=='single'):
            label = torch.cat((label[:,[0]],label[:,[0]]),dim=1)

        return mixture, mix_name, label, mix_way[0]
        
