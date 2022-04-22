# DenseBiasNet-pytorch
The dense biased network implementated by pytorch has two versions:

1. DenseBiasNet_a.py, DenseBiasNet_a_deeper.py, DenseBiasNet_state_a and DenseBiasNet_state_a_deeper.py are implementated via the dense biased connection in our MICCAI version.

2. DenseBiasNet_state_b.py, DenseBiasNet_state_b_deeper.py and DenseBiasNet_b.py are implementated via the dense biased connection in our MedIA version.

When 'state' in the file name, the bias quantity $m$ in the dense bias connection is the same fixed value such as 1, 2, etc in each layer. When there is no 'state' in file name, the bias quantity $m$ in the dense bias connection is the proportion of the feature maps from each layer.

If you use densebiasnet or some part of the code, please cite (see bibtex):
* MICCAI version:
**DPA-DenseBiasNet: Semi-supervised 3D Fine Renal Artery Segmentation with Dense Biased Network and Deep Priori Anatomy**  [https://doi.org/10.1007/978-3-030-32226-7_16](https://doi.org/10.1007/978-3-030-32226-7_16) 

* MedIA version:
**Dense biased networks with deep priori anatomy and hard region adaptation: Semi-supervised learning for fine renal artery segmentation**
[https://doi.org/10.1016/j.media.2020.101722](https://doi.org/10.1016/j.media.2020.101722)
