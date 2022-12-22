import os
import sys
import math
import random
import shutil
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import soundfile as sf

from sms_wsj.database.create_rirs import config, scenarios, rirs
from sms_wsj.reverb.reverb_utils import convolve


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
        y: (3,2)
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

  
class Dataset(data.Dataset):
    def __init__(
        self,
        reverb_matrixs_dir,
        num_rir = 100,
        num_utt = 100,
        segment = 8,
        channel = [0,1,2,3],
        overlap = [0.0, 0.1, 0.2, 0.3, 0.4,],
        overlap_type = 'single',
        snr_low = 5,
        snr_high = 25,
        raw_dir = '/path/to/LibriSpeech/filelist-all/',
        noise_dir = '/path/to/noise/',
    ):
        super(Dataset, self).__init__()
        self.reverb_matrixs_dir = reverb_matrixs_dir
        self.num_rir = num_rir
        self.num_utt = num_utt
        self.segment = segment
        self.channel = channel
        self.overlap = overlap
        self.overlap_type = overlap_type
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.raw_dir = raw_dir
        self.noise_wav = noise_wav

    def __len__(self):
        return self.num_utt
      
    def add_reverb(self, raw_dir1, raw_dir2, h_use):
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

                mix_way = np.random.choice([self.overlap_type], size=1, replace=False)

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
      
    def add_noise(self, mix_reverb, w_n):
        SNR = random.uniform(self.snr_low, self.snr_high)
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
        num_spk = len(raw_list)

        speaker1 = np.random.randint(0, num_spk)
        speaker2 = np.random.randint(0, num_spk)
        while (speaker1 == speaker2):
            speaker2 = np.random.randint(0,num_spk)
        raw_dir1 = self.raw_dir+raw_list[speaker1]
        raw_dir2 = self.raw_dir+raw_list[speaker2]

        choose_rir = np.random.randint(0, self.num_rir)
        rand_rir = np.load(self.reverb_matrixs_dir + str(choose_rir).zfill(4) + '.npz')
        h_use, _source_positions, _sensor_positions, room_dimensions, sound_decay_time, = \
                rand_rir['h'], rand_rir['source_positions'], rand_rir['sensor_positions'], rand_rir['room_dimensions'], rand_rir['sound_decay_time']

        mix_reverb, s1_reverb, s2_reverb, mix_name, mix_way = self.add_reverb(raw_dir1,raw_dir2,h_use[:,self.channel,:])

        w_n, _ = sf.read(self.noise_wav, dtype="float32")
        mix_noise = self.add_noise(mix_reverb, w_n)
        mix_noise = mix_noise.transpose()

        mixture = torch.from_numpy(np.array(mix_noise).astype(np.float32)).permute(1,0)

        _source_positions = _source_positions[...,None]               # (3,src,channel)
        _sensor_positions = _sensor_positions[:,self.channel][:,None] # (3,src,channel)
        # 3d-to-2d 2021.09.21
        _source_positions[2] = _sensor_positions[2,0,0]

        label = get_label(_source_positions, _sensor_positions, usage='simu')   # (3,src) 
        label = torch.from_numpy(label.astype(np.float32))

        if (mix_way[0]=='single'):
            label = torch.cat((label[:,[0]],label[:,[0]]),dim=1)

        return mixture, s1_reverb, s2_reverb, mix_name, label, mix_way[0], _source_positions, _sensor_positions, room_dimensions, sound_decay_time
      
      
if __name__ == "__main__":
    num_rir, num_utt = 1, 1
    rir_dir = 'path/to/testset'
    wav_dir = os.path.join(rir_dir, 'mix_noise')
    reverb_matrixs_dir = os.path.join(rir_dir, 'reverb_matrixs')
    s1_dir = os.path.join(rir_dir, 's1')
    s2_dir = os.path.join(rir_dir, 's2')

    for item in [wav_dir, reverb_matrixs_dir, s1_dir, s2_dir]:
        try:
            os.makedirs(item)
        except OSError:
            pass

    overlap_pattern = ['dominant']
    d = Dataset(reverb_matrixs_dir = '/path/to/reverb-set/',
                num_rir = num_rir,
                num_utt = num_utt,
                overlap_type = overlap_pattern[0],
                snr_low = 5,
                snr_high = 25,)

    print('saving reverb matrixs into', reverb_matrixs_dir)
    pbar = tqdm(range(num_utt))
    for i in pbar:
        mixture, s1, s2, mix_name, label, mix_way, source_positions, sensor_positions, room_dimensions, sound_decay_time = d[i]
        sf.write(os.path.join(wav_dir, mix_name), mixture.numpy(), 16000)
        sf.write(os.path.join(s1_dir, mix_name), s1[:,0], 16000)
        sf.write(os.path.join(s2_dir, mix_name), s2[:,0], 16000)
        np.savez(os.path.join(reverb_matrixs_dir, mix_name[:-4] + '.npz'), \
                mix=mixture, n=mix_name, label=label, mix_way=mix_way, source_positions=source_positions, sensor_positions=sensor_positions, \
                room_dimensions=room_dimensions, sound_decay_time=sound_decay_time)

    print('Done.')
