# MTKD

This code is built based on [**BasicSR**](https://github.com/XPixelGroup/BasicSR), where more information can be found to help use this code.

Thanks to [**pytorch_wavelets**](https://github.com/fbcotter/pytorch_wavelets.git) for Wavelet Transforms in Pytorch.

## Installation
Please refer to [**BasicSR**](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md)

## Preparing datasets
### Training sets:
[[DIV2K]](https://data.vision.ee.ethz.ch/cvl/DIV2K/) More details are in [DatasetPreparation.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#image-super-resolution)
### Test sets: 
[[Set5]](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

[[Set14]](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

[[BSD100]](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

[[urban100]](https://sites.google.com/site/jbhuang0604/publications/struct_sr)


## Model ZOO
[**EDSR Official Models**](https://drive.google.com/drive/folders/1rtJCHuOAEixB1OWmUVbbVm158vzC3kTt)

[**SwinIR Official Models**](https://github.com/JingyunLiang/SwinIR/releases)

[**RCAN Official Models**](https://drive.google.com/file/d/10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa/view)

Our trained model will be published soon!!

## Use
Step 1 - Train DCTSwin
```
python .\basicsr\train.py -opt .\options\train\ECCVMTKD\DCTSwinx4.yml
```
Step 2 - Train EDSRbl
```
python .\basicsr\train.py -opt .\options\train\ECCVMTKD\train_EDSR_Mx4_KD.yml
```


## Citation
```
@article{jiang2024mtkd,
  title={MTKD: Multi-Teacher Knowledge Distillation for Image Super-Resolution},
  author={Jiang, Yuxuan and Feng, Chen and Zhang, Fan and Bull, David},
  journal={arXiv preprint arXiv:2404.09571},
  year={2024}
}
```
