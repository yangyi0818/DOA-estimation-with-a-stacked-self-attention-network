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


[Paper]: https://doi.org/10.1121/10.0016467
