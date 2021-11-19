
# World Wide Mind
Real-time EEG decoding across 100 brains. A poster presentation for this project can be found [here](https://jamestev.github.io/msc-dissertation-poster.pdf) and the full report [here](https://jamestev.github.io/msc-dissertation.pdf).

## Setup 
See [MicroPython setup](/micropython/README.md) for installing, building and flashing a port of MicroPython for the Espressif ESP32 used in this project. Instructions are also provided for development and experimentation.
## Background

This project formed part of an MSc dissertation in collaboration with the [Next Generation Neural Interfaces (NGNI) Lab](https://www.imperial.ac.uk/next-generation-neural-interfaces) at Imperial College London. It was intended to explore the possibility of simultaneous decoding and visualisation of EEG signals acquired from ~100 different audience members simultaneously during a large scale exhibition. Signals acquired from the BCI devices are used for collaborative control in a multiplayer game (using only mental control). 

The cost of existing BCI technologies makes them inaccessible to the general public and prohibits their use on a mass scale. This project aimed to create a __novel, ultra low-cost BCI prototype__ that can change this in the hope to increase public engagement and facilite education in the field of neurotechnologies.

## Objectives
The core focus of this study is to develop real time decoding and communication of raw EEG signals acquired from a proprietary EEG hardware device developed by the NGNI Lab.

### Constraints
- very tight budget of ~ £20 per device
- all processing related to sampling, signal processing,
decoding and networking must be performed on-device
- use the NGNI hardware prototype based on the Espressif
ESP32 SoC (Tensilica Xtensa LX6 MCU)
- real-time decoding and communication to an AWS cloud
service
- non-invasive BCI using only ‘dry’ surface electrodes

## Design

### SSVEPs for BCI control
The core role of a BCI is to interpret intentions of a user by making sense of their brain signals. Steady state visual evoked potentials (SSVEPs) are modulations in the brain’s visual cortex in response to a visual stimulus which can be measured as sinusoids at the frequency of the visual stimulus being observed. Visual stimuli usually take the form of shapes that flicker at predetermined frequencies.

A very simple SSVEP interface with flickering squares is provided in `ui/ssvep_squares.html`. Simply open it with your browser to try it out. You can use the url query parameters to adjust flicker frequencies. For example, by modifying the url in the browser to something like `<path-to-file>?up=10&right=12` would set the upper square to flicker at 10Hz and the right at 12Hz. Note that these frequencies are approximate and depend largely on the browser you're using and the load on your machine. 

### SSVEP decoding
The EEG literature widely reports that multivariate statistical techniques - such as canonical correlation analysis (CCA) and its extensions - are optimal for SSVEP decoding. This project primarily explored two extensions: Multi-setCCA (MsetCCA) [1] and Generalised CCA (GCCA) [2]. An implementation of the seminal TRCA algorithm from [3] is also investigated.

## Performance
Results showed that the MsetCCA was most effective and could achieve accuracy and ITR rates comparable with BCIs in the literature. Two promising parameter combinations:
- 4 calibration trials of 0.75s each: accuracy of 95.56 ± 3.74% with ITR of 102 bits/min
- 2 calibration trials of 1s each: accuracy of 80.56 ± 4.46% with ITR of 40 bits/min

See the [full report](https://jamestev.github.io/msc-dissertation.pdf) for more detailed explanations, results and analysis.
## References
[1] Y. Zhang, G. Zhou, J. Jin, X. Wang, and A. Cichocki, “Frequency recognition in ssvep-based bci using multiset canonical correlation analysis,” International journal of neural systems, vol. 24, no. 04, p. 1 450 013, 2014.

[2] Q. Sun, M. Chen, L. Zhang, X. Yuan, and C. Li, “Improving ssvep identification accuracy via generalized canonical correlation analysis,” in 2021 10th International IEEE/EMBS Conference on Neural Engineering (NER), IEEE, 2021, pp. 61–64.

[3] H. Tanaka, T. Katura, and H. Sato, “Task-related component analysis for functional neuroimaging and application to near-infrared spectroscopy data,” NeuroImage, vol. 64, pp. 308–327, 2013.