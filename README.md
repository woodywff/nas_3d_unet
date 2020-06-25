# NAS-3D-U-Net

## Introduction
This is the source code of this article: [Neural Architecture Search for Gliomas Segmentation on Multimodal Magnetic Resonance Imaging](https://arxiv.org/abs/2005.06338)

`Main.ipynb` is the main storyline which is self-explained. (If you don't wanna run it in Jupyter Notebook, don't forget to change the `tqdm.notebook` to `tqdm`.)

Most of the configurations are set up in `config.yml`.

I've put the three downloaded and unzipped dataset files in a `nas_3d_unet/data/` folder as
```
nas_3d_unet/data/MICCAI_BraTS_2019_Data_Training/
nas_3d_unet/data/MICCAI_BraTS_2019_Data_Validation/
nas_3d_unet/data/MICCAI_BraTS_2019_Data_Testing/
```
You are free to keep these things anywhere else, just don't forget to change the corresponding arguments in the `config.yml`.

This repository is also an update for the previous [brats2019 pipeline](https://github.com/woodywff/brats_2019-data_pipeline).

## Development Environment
CUDA10 torch==1.2.0 torchvision==0.4.0

GTX1060 (6GB GPU Memory) is good enough for running the whole project (both searching and training) with patchsize=64.

GTX1080Ti (11GB GPU Memory) is recommended.

## Acknowledgment
- This work refers a lot to [tianbaochou/NasUnet](https://github.com/tianbaochou/NasUnet) and [ellisdg/3DUnetCNN](https://github.com/ellisdg/3DUnetCNN). We deeply appreciate their contributions to the community.

- Many thanks to [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019.html).



