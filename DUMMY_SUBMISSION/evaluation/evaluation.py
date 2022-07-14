from os import listdir
from os.path import join
import SimpleITK as sitk
import numpy as np
import pandas as pd

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

def dice(pre, gt):
    tmp = pre + gt
    a = np.sum(np.where(tmp == 2, 1, 0))
    b = np.sum(pre)
    c = np.sum(gt)
    dice = (2*a)/(b+c)
    return dice

def Getcontour(img):

    image = sitk.GetImageFromArray(img.astype(np.uint8), isVector=False)

    filter = sitk.SimpleContourExtractorImageFilter()
    image = filter.Execute(image)
    image = sitk.GetArrayFromImage(image)
    return image.astype(np.uint8)

def HausdorffDistance(predict, label, index=1):
    predict = (predict == index).astype(np.uint8)
    label = (label == index).astype(np.uint8)
    predict_sum = predict.sum()
    label_sum = label.sum()
    if predict_sum != 0 and label_sum != 0 :
        mask1 = sitk.GetImageFromArray(predict,isVector=False)
        mask2 = sitk.GetImageFromArray(label,isVector=False)
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(mask1, mask2)
        result1 = hausdorff_distance_filter.GetHausdorffDistance()
        result2 = hausdorff_distance_filter.GetAverageHausdorffDistance()
        result = result1,result2
    elif predict_sum != 0 and label_sum == 0:
        result = 'FP','FP'
    elif predict_sum == 0 and label_sum != 0:
        result = 'FN','FN'
    else:
        result = 'TN','TN'
    return result

def DSC(model_name, n_classes, pred_dir, gt_dir):
    pred_filenames = listdir(join(pred_dir, model_name))
    DSC = np.zeros((n_classes, len(pred_filenames)), dtype=np.float32)

    save_name = model_name+'_DSC.csv'

    for i in range(len(pred_filenames)):
        name = pred_filenames[i]

        predict = sitk.ReadImage(join(pred_dir, model_name, name))
        predict = sitk.GetArrayFromImage(predict)
        predict = to_categorical(predict, num_classes=n_classes)

        groundtruth = sitk.ReadImage(join(gt_dir, name))
        groundtruth = sitk.GetArrayFromImage(groundtruth)
        groundtruth = to_categorical(groundtruth, num_classes=n_classes)

        for c in range(n_classes):
            DSC[c, i] = dice(predict[c], groundtruth[c])

        print(name, DSC[1:, i])

    df = pd.DataFrame({'Name': pred_filenames, 'Vein': DSC[1], 'Kidney': DSC[2], 'Artery': DSC[3], 'Tumor': DSC[4]})
    df.to_csv(save_name, index=False)

def HDAVD(model_name, n_classes, pred_dir, gt_dir):
    pred_filenames = listdir(join(pred_dir, model_name))
    hauAve = np.zeros(shape=(n_classes, len(pred_filenames)), dtype=np.float32)
    hau = np.zeros(shape=(n_classes, len(pred_filenames)), dtype=np.float32)

    save_name_HD = model_name+'_HD.csv'
    save_name_AVD = model_name+'_AVD.csv'

    for i in range(len(pred_filenames)):
        name = pred_filenames[i]

        groundtruth = sitk.ReadImage(join(gt_dir, name))
        originSpacing = groundtruth.GetSpacing()
        originSize = groundtruth.GetSize()
        newSize = [int(round(originSize[0] * originSpacing[0])), int(round(originSize[1] * originSpacing[1])),
                   int(round(originSize[2] * originSpacing[2]))]
        newSpacing = [1, 1, 1]

        predict = sitk.ReadImage(join(pred_dir, model_name, name))
        predict.SetSpacing([originSpacing[0], originSpacing[0], originSpacing[0]])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(groundtruth)
        resampler.SetSize(newSize)
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        groundtruth = resampler.Execute(groundtruth)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(predict)
        resampler.SetSize(newSize)
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        predict = resampler.Execute(predict)

        predict = sitk.GetArrayFromImage(predict)
        predict = to_categorical(predict, num_classes=n_classes)

        groundtruth = sitk.GetArrayFromImage(groundtruth)
        groundtruth = to_categorical(groundtruth, num_classes=n_classes)
        for c in range(n_classes):
            predict_suf = Getcontour(predict[c])
            label_suf = Getcontour(groundtruth[c])
            HD_AVD = HausdorffDistance(predict_suf, label_suf)
            if HD_AVD[0] == 'FN' or HD_AVD[0] == 'FP' or HD_AVD[0] == 'TN':
                predict_suf = np.zeros_like(predict[c])
                predict_suf[0, :, :] = 1.
                predict_suf[-1, :, :] = 1.
                predict_suf[:, 0, :] = 1.
                predict_suf[:, -1, :] = 1.
                predict_suf[:, :, 0] = 1.
                predict_suf[:, :, -1] = 1.
                print("The " + str(c) + "th structure fails to be segmented, so the image boundary is used as the evaluation position of this structure.")

                hau[c, i], hauAve[c, i] = HausdorffDistance(predict_suf, label_suf)
            else:
                hau[c, i], hauAve[c, i] = HD_AVD[0], HD_AVD[1]

        print(name, hau[1:, i], hauAve[1:, i])

    df_AVD = pd.DataFrame({'Name': pred_filenames, 'Vein': hauAve[1], 'Kidney': hauAve[2], 'Artery': hauAve[3], 'Tumor': hauAve[4]})
    df_AVD.to_csv(save_name_AVD, index=False)
    df_HD = pd.DataFrame({'Name': pred_filenames, 'Vein': hau[1], 'Kidney': hau[2], 'Artery': hau[3], 'Tumor': hau[4]})
    df_HD.to_csv(save_name_HD, index=False)
