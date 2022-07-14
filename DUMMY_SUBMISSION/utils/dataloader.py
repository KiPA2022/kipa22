from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np

from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, file_dir, shape, num_classes, is_aug=True):
        super(DatasetFromFolder3D, self).__init__()
        self.is_aug = is_aug
        self.image_filenames = [x for x in listdir(join(file_dir, "image")) if is_image_file(x)]
        self.file_dir = file_dir
        self.shape = shape
        self.num_classes = num_classes
        if is_aug:
            self.mirror_transform = MirrorTransform(axes=(0, 1, 2))
            self.spatial_transform = SpatialTransform(patch_size=shape,
                                                      patch_center_dist_from_border=np.array(shape)//2,
                                                      do_elastic_deform=True,
                                                      alpha=(0., 1000.),
                                                      sigma=(10., 13.),
                                                      do_rotation=True,
                                                      angle_x=(-np.pi / 9, np.pi / 9),
                                                      angle_y=(-np.pi / 9, np.pi / 9),
                                                      angle_z=(-np.pi / 9, np.pi / 9),
                                                      do_scale=True,
                                                      scale=(0.75, 1.25),
                                                      border_mode_data='constant',
                                                      border_cval_data=0,
                                                      order_data=1,
                                                      random_crop=True)
        else:
            self.spatial_transform = SpatialTransform(patch_size=shape,
                                                      patch_center_dist_from_border=np.array(shape) // 2,
                                                      do_elastic_deform=False,
                                                      do_rotation=False,
                                                      do_scale=False,
                                                      scale=(0.75, 1.25),
                                                      border_mode_data='constant',
                                                      border_cval_data=0,
                                                      order_data=1,
                                                      random_crop=True)

    def __getitem__(self, index):
        image = sitk.ReadImage(join(self.file_dir, "image", self.image_filenames[index]))
        image = sitk.GetArrayFromImage(image)
        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image.astype(np.float32)
        image = image / 2048.

        target = sitk.ReadImage(join(self.file_dir, "label", self.image_filenames[index]))
        target = sitk.GetArrayFromImage(target)
        target = target.astype(np.float32)

        image, target = self.pad(image, target, self.shape)

        image = image[np.newaxis, np.newaxis, :, :, :]
        target = target[np.newaxis, np.newaxis, :, :, :]
        data_dict = {"data": image,
                     "seg": target}
        if self.is_aug:
            data_dict = self.mirror_transform(**data_dict)

            data_dict = self.spatial_transform(**data_dict)
        else:
            data_dict = self.spatial_transform(**data_dict)

        image = data_dict.get("data")
        target = data_dict.get("seg")
        target = self.to_categorical(target[0, 0], self.num_classes)
        target = target.astype(np.float32)
        image = image[0]
        return image, target

    def pad(self, image, label, croped_shape):
        if image.shape[0] < croped_shape[0]:
            zero = np.zeros((croped_shape[0] - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
            image = np.concatenate([image, zero], axis=0)
            label = np.concatenate([label, zero], axis=0)
        if image.shape[1] < croped_shape[1]:
            zero = np.zeros((image.shape[0], croped_shape[1] - image.shape[1], image.shape[2]), dtype=np.float32)
            image = np.concatenate([image, zero], axis=1)
            label = np.concatenate([label, zero], axis=1)
        if image.shape[2] < croped_shape[2]:
            zero = np.zeros((image.shape[0], image.shape[1], croped_shape[2] - image.shape[2]), dtype=np.float32)
            image = np.concatenate([image, zero], axis=2)
            label = np.concatenate([label, zero], axis=2)

        return image, label


    def to_categorical(self, y, num_classes=None):
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

    def __len__(self):
        return len(self.image_filenames)

