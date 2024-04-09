# IFViT: Interpretable Fixed-Length Representation for Fingerprint Matching via Vision Transformer
<p align="center"><img src="figs/FVIT Structure.jpg" width="1000"/></p>

## IFViT
### Abstract
Determining dense feature points on fingerprints used in constructing deep fixed-length representations for accurate matching, particularly at the pixel level, is of significant interest. To explore the interpretability of  fingerprint matching, we propose a multi-stage interpretable fingerprint matching network, namely Interpretable Fixed-length Representation for Fingerprint Matching via Vision Transformer (IFViT), which consists of two primary modules. The first module, an interpretable dense registration module, establishes a Vision Transformer (ViT)-based Siamese Network to capture long-range dependencies and the global context in fingerprint pairs. It provides interpretable dense pixel-wise correspondences of feature points for fingerprint alignment and enhances the interpretability in the subsequent matching stage. The second module takes into account both local and global representations of the aligned fingerprint pair to achieve an interpretable fixed-length representation extraction and matching. It employs the ViTs trained in the first module with the additional fully connected layer and retrains them to simultaneously produce the discriminative fixed-length representation and interpretable dense pixel-wise correspondences of feature points. Extensive experimental results on diverse publicly available fingerprint databases demonstrate that the proposed framework not only exhibits superior performance on dense registration and matching but also significantly promotes the interpretability in deep fixed-length representations-based fingerprint matching.
### Demonstration
<p align="center"><img src="figs/Samples of registration.jpg" width="1000"/></p>
<p align="center"><img src="figs/Samples of intepretability.jpg" width="1000"/></p>

## Requirements
* python==3.8.10
* accelerate==0.20.3
* einops==0.7.0
* kornia==0.6.9
* matplotlib==3.5.0
* numpy==1.21.1
* opencv-python==4.6.0.66
* pytorch-metric-learning==2.3.0
* torch==1.13.1
* torchvision==0.14.1
* tqdm==4.63.0
* vit-pytorch==1.6.5

## Remark
Since our work has been submitted and is currently under review at the IEEE Transactions on Information Forensics and Security (TIFS), only partial non-core codes have been released for verification purposes. The complete code will be organized and made available upon the acceptance and publication of this work.

## Acknowledgement
In terms of the reproduction of DeepPrint, we referred to and modified the implementation provided by [fixed-length-fingerprint-extractors](https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors). Many thanks for their wonderful work.
