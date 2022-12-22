import os
import sys
import numpy as np
import torch
import random
import time
from multiprocessing import Pool

from sms_wsj.database.create_rirs import config, scenarios, rirs
from sms_wsj.reverb.reverb_utils import convolve

T60_LOW, T60_HIGH = 0.15, 0.60

def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)
    
def generate_rir(i):
    _worker_init_fn_(i)
    reverb_matrixs_dir = '/path/to/reverb-set/'
    geometry, sound_decay_time_range, sample_rate, filter_length = config(T60_LOW, T60_HIGH)
    room_dimensions, source_positions, sensor_positions, sound_decay_time = scenarios(geometry, sound_decay_time_range,)
    h = rirs(sample_rate, filter_length, room_dimensions, source_positions, sensor_positions, sound_decay_time)
    np.savez(reverb_matrixs_dir + str(i).zfill(4) + '.npz', h=h, source_positions=source_positions, sensor_positions=sensor_positions,
             room_dimensions=room_dimensions, sound_decay_time=sound_decay_time,)

if __name__ == "__main__":
    nj = 32
    num_rir = 3000
    reverb_matrixs_dir = '/path/to/reverb-set/'

    if not os.path.exists(reverb_matrixs_dir):
        os.makedirs(reverb_matrixs_dir)
    else:
        if (input('target dir already esists, continue? [y/n]  ') == 'n'):
            print('Exit. Nothing happends.')
            sys.exit()
    print('Generating reverb matrixs into ', reverb_matrixs_dir, '......')

    time_start=time.time()
    pool = Pool(processes=nj)
    args = []
    for i in range (num_rir):
        args.append(i)
    pool.map(generate_rir, args)
    pool.close()
    pool.join()
    time_end=time.time()
    print('totally cost ', round((time_end-time_start)/60), 'minutes')
