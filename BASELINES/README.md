# Baselines

This is a depository for the officially implemented baseline models and the connections of excellent segmentation framework published by third parties.

## Officially implemented baselines
The baselines in this folder will be gradually improved, and the existing codes include:
- [DenseBiasNet](https://github.com/YutingHe-list/DenseBiasNet-pytorch) and its variants [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300864)]
- [MNet](https://github.com/zfdong-code/MNet)
- [3D U-Net](https://github.com/KiPA2022/kipa22/blob/main/BASELINES/models/UNet3D.py)[[paper](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)]
- ...

The performance of the baselines on the open test set of KiPA22.

|Baselines    |Kidney DSC |Kidney HD |Kidney AVD |Tumor DSC  |Tumor HD  |Tumor AVD  |Artery DSC |Artery HD |Artery AVD |Vein DSC |Vein HD |Vein AVD |
|:------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---:|:---:|:---:|
|DenseBiasNet |94.6   |23.89  |0.79   |79.3   |27.97  |4.33   |84.5   |26.67  |1.31   |76.1 |34.60|2.08 |
|MNet         |90.6   |44.03  |2.16   |65.1   |61.05  |10.23  |78.2   |47.79  |2.71   |73.5 |42.60|3.06 |
|3D U-Net     |91.7   |18.44  |0.75   |66.6   |24.02  |4.45   |71.9   |22.17  |1.10   |60.9 |22.26|3.37 |

## Enviroment and requirement
- python 3.7+
- SimpleITK 2.0.2
- Pytorch 1.7+
- batchgenerators 0.23

## Third party baselines
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [V-Net](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/vnet.py)
- [DeepLab](https://github.com/jfzhang95/pytorch-deeplab-xception)
