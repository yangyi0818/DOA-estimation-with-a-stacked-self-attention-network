# DOA-estimation-with-a-stacked-self-attention-network
**A stacked self-attention network for two-dimensional direction-of-arrival estimation in hands-free speech communication**

**This paper has been accepted by *Journal of the Acoustical Society of America (JASA).*  Available [here][Paper].**

## Contents 
* **[DOA-estimation-with-a-stacked-self-attention-network](#doa-estimation-with-a-stacked-self-attention-network)**
  * **[Contents](#contents)**
  * **[Introduction](#introduction)**
  * **[Dataset](#dataset)**
  * **[Requirement](#requirement)**
  * **[Train](#train)**
  * **[Test](#test)**
  * **[Results](#results)**
  * **[Citation](#citation)**
  * **[References](#references)**

## Introduction
**When making voice interactions with hands-free speech communication devices, direction-of-arrival estimation is an essential step. To address the detrimental influence of unavoidable background noise and interference speech on direction-of-arrival estimation, we introduce a stacked self-attention network system, a supervised deep learning method that enables utterance level estimation without requirement for any pre-processing such as voice activity detection. Specifically, alternately stacked time- and frequency-dependent self-attention blocks are designed to process information in terms of time and frequency, respectively. The former blocks focus on the importance of each time frame of the received audio mixture and perform temporal selection to reduce the influence of non-speech and interference frames, while the latter blocks are utilized to derive inner-correlation among different frequencies. Additionally, the non-causal convolution and self-attention networks are replaced by causal ones, enabling real-time direction-of-arrival estimation with a latency of only 6.25 ms. Experiments with simulated and measured room impulse responses, as well as real recordings, verify the advantages of the proposed method over the state-of-the-art baselines.**

![image](https://github.com/yangyi0818/DOA-estimation-with-a-stacked-self-attention-network/blob/main/figures/model-architecture1.png)
![image](https://github.com/yangyi0818/DOA-estimation-with-a-stacked-self-attention-network/blob/main/figures/model-architecture2.png)

## Dataset
**We use [sms_wsj][sms_wsj] to generate room impulse responses (RIRs) set. ```sms_wsj/reverb/scenario.py``` and ```sms_wsj/database/create_rirs.py``` should be replaced by scripts in 'sms_wsj_replace' folder.**

**use ```python generate_rir.py``` to generate training and valadation data**

## Requirement
**Our script use [asteroid][asteroid] toolkit as the basic framework.**

## Train
**We recommend running to train end-to-end :**

**```./run.sh --id 0,1,2,3```**

**or :**

**```./run.sh --id 0,1,2,3 --stage 1```**

## Test
**```./run.sh --id 0 --stage 2```**

## Results
**The average MAE (degree), Acc. (%), model parameters, and latency of the real-time implementation of the proposed system and the CNN-baseline [1] on SS condition of all simulated test sets. (E_theta = 15 degree).**

|**Measure** |**MAE** |**Acc.**|**Parameters**|**Latency**|
| :-----     | :----: | :----: | :----:       | :----:    |
|**CNN [1]** |3.6     |99.3    |8.7M          |14 ms      |
|**Proposed**|2.9     |99.5    |282k          |6.25 ms    |

**The average MAE (degree) and Acc. (%) of the off-line and real-time implementations of the proposed system for each overlap condition on all simulated test sets. (E_theta = 15 degree).**

|**Overlap condition**|**SS**  |**SS**  |**IO**  |**IO**  |**PO**  |**PO**  |
| :-----              | :----: | :----: | :----: | :----: | :----: | :----: |
|**Measure**          |**MAE** |**Acc.**|**MAE** |**Acc.**|**MAE** |**Acc.**|
|**Off-line**         |4.3     |97.4    |8.3     |88.8    |7.2     |91.1    |
|**Real-time**        |5.2     |95.8    |8.6     |86.5    |9.0     |84.6    |

**An attention map of an example speech utterance (room dimension = 7.0 m × 6.0 m × 3.2 m, RT60 = 400 ms, SNR = 20 dB, SIR = 0dB). The ground-truth and estimated azimuths for the target speaker (speaker A) are 1.6 degree and 2.3 degree, respectively. The ground-truth azimuth for the interference speaker (speaker B) is 125.6 degree. The horizontal and vertical axes represent the frame index of interest and the frames to which it attends. The log power spectrums of the input mixture, reverberated utterances of speaker A and speaker B are also attached on the top and left, respectively.**

![image](https://github.com/yangyi0818/DOA-estimation-with-a-stacked-self-attention-network/blob/main/figures/attention-map.png)

**The off-line and real-time 2-D DOA estimation curves for each overlap condition.**

![image](https://github.com/yangyi0818/DOA-estimation-with-a-stacked-self-attention-network/blob/main/figures/real-time-curve-SS.png)
![image](https://github.com/yangyi0818/DOA-estimation-with-a-stacked-self-attention-network/blob/main/figures/real-time-curve-IO.png)
![image](https://github.com/yangyi0818/DOA-estimation-with-a-stacked-self-attention-network/blob/main/figures/real-time-curve-PO.png)

## Citation
Cite our paper by:  

@article{yang2022stacked,  

  title={A stacked self-attention network for two-dimensional direction-of-arrival estimation in hands-free speech communication},  
  
  author={Yang, Yi and Chen, Hangting and Zhang, Pengyuan},  
  
  journal={The Journal of the Acoustical Society of America},  
  
  volume={152},  
  
  number={6},  
  
  pages={3444--3457},  
  
  year={2022},  
  
  publisher={Acoustical Society of America}  
  
}

## References
[1] A. Kucuk, A. Ganguly, Y. Hao, and I. M. S. Panahi, "Real-time convolutional neural network-based speech source localization on smartphone," IEEE Access 7, 169969–169978 (2019).

**Please feel free to contact us if you have any questions.**

[Paper]: https://doi.org/10.1121/10.0016467
[sms_wsj]: https://github.com/fgnt/sms_wsj
[asteroid]: https://github.com/asteroid-team/asteroid

