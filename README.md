# Official repository of MICCAI 2022 Challenge: [KiPA22](https://kipa22.grand-challenge.org/).

We provide the evaluation code and a baseline depository for a easy start of our challenge. You can use the code in [`EVALUATION`](https://github.com/KiPA2022/kipa22/tree/main/EVALUATION) to evaluate the `DSC`, `HD`, and `AVD` of your result, and use the code in [`BASELINES`](https://github.com/KiPA2022/kipa22/tree/main/BASELINES) to strat your KiPA22 easily.

Please feel free to raise any issues if you have questions about the challenge, e.g., dataset, evaluation measures, ranking scheme and so on.

<p align="center"><img width="70%" src="figs/intro.png" /></p>

## EVALUATION
The `EVALUATION` folder provides the official implementation of the [evaluation metrics](https://kipa22.grand-challenge.org/evaluation-details/) used in the KiPA22 Challenge, including:
- `DSC` (Dice Similarity Coefficient) is used to evaluate the area-based overlap index. 
- `HD` (Hausdorff Distance) is used to compare the segmentation quality of outliers.
- `AVD` (Average Hausdorff Distance) is used to evaluate the coincidence of the surface for stable and less sensitive to outliers.

More details are available in the papers ([He et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S1361841521001018); [Taha et al., 2015](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x)).

## BASELINES
The `BASELINES` folder provides some basic models of official implementation and the connections of some excellent third-party libraries which will help the participants start their training easily. The baselines in this folder will be gradually improved, and the existing codes include:
- The [code](https://github.com/YutingHe-list/DenseBiasNet-pytorch) of [DenseBiasNet and its variants](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300864)
- ...
