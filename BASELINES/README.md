# Baselines

This is a depository for the officially implemented baseline models and the connections of excellent segmentation framework published by third parties.

## Officially implemented baselines
The baselines in this folder will be gradually improved, and the existing codes include:
- [DenseBiasNet](https://github.com/YutingHe-list/DenseBiasNet-pytorch) and its variants [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300864)]
- [MNet](https://github.com/zfdong-code/MNet)
- [3D U-Net](https://github.com/KiPA2022/kipa22/blob/main/BASELINES/models/UNet3D.py)[[paper](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)]
- ...

The performance of the baselines on the open test set of KiPA22.

|   Baselines | Kidney   | Kidney    |  Kidney |  Tumor |   Tumor |  Tumor |   Artery |   Artery | Artery |Vein|Vein|Vein|
|--------:|--------:|------:|--------:|-------------:|--------------:|---------------:|-----------------:|--------------:|
|DenseBiasNet | 1X       | False |    1.28 |          |          |           6.50 |             | |1.95 |          | | 3.36|

## Enviroment and requirement
- python 3.7+
- SimpleITK 2.0.2
- Pytorch 1.7+
- batchgenerators 0.23

## Third party baselines
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [V-Net](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/vnet.py)
- [DeepLab](https://github.com/jfzhang95/pytorch-deeplab-xception)
