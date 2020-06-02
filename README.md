# NAS-3D-U-Net

## Introduction
This is the source code of this article: [Neural Architecture Search for Gliomas Segmentation on Multimodal Magnetic Resonance Imaging](https://arxiv.org/abs/2005.06338)

`Main.ipynb` is the main storyline which is self-explained. (If you don't wanna run it in Jupyter Notebook, don't forget to change the `tqdm.notebook` to `tqdm`.)

Most of the configurations are set up in `config.yml`.

This repository is also an update for the previous [brats2019 pipeline](https://github.com/woodywff/brats_2019-data_pipeline).

## Development Environment
CUDA10 torch==1.2.0 torchvision==0.4.0

GTX1060 (6GB GPU Memory) is good enough for running the whole project (both searching and training) with patchsize=64.

GTX1080Ti (11GB GPU Memory) is recommended.

## Acknowledgment
- Most of the NAS code in this work refers a lot to [NasUnet](https://github.com/tianbaochou/NasUnet). We deeply appreciate their contributions to the community.

- Many thanks to [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019.html).



