import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models

import re
import os
from glob import glob
import numpy as np
from PIL import Image, ImageOps


def get_random_crop(image, crop_height, crop_width):
    """get random crop dimensions"""

    image = np.asarray(image)

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    left = x
    top = y
    right = x + crop_width
    bottom = y + crop_height
    
    return (left, top, right, bottom)


def expand_training_data(path, expand_factor=1024, crop_height=224, crop_width=224):
    """
    'This increases the size of ourtraining set by a factor of 2048, 
    though the resulting training examples are, of course, 
    highly inter-dependent.'

    Since reflections will double the count of training data,
    expanding_factor is set to half of 2048
    """

    for img in glob(os.path.join(path,'*')):
        file_name = re.search(r'(\w+)\.jpg',img).group()
        ext = '.jpg'
        image = Image.open(img)
        for i in range(expand_factor):
            crop_dims = get_random_crop(image, crop_height, crop_width)
            transformed_img = image.crop(crop_dims)
            mirrored_transformed_img = ImageOps.mirror(transformed_img)
            transformed_img.save(
                os.path.join(path,file_name+i+ext), 
                quality=95)
            mirrored_transformed_img.save(
                os.path.join(path,file_name+i+'flipped'+ext), 
                quality=95)


class PCA_normalize(object):
    """
    PCA transform on image

    https://datascience.stackexchange.com/questions/30602/
    how-to-implement-pca-color-augmentation-as-discussed-in-alexnet
    """

    def __call__(self, image):
        image = np.asarray(image, dtype='float32')
        image = np.reshape(image,(image.shape[0]*image.shape[1],3))

        mean = np.mean(image, axis=0)
        std = np.std(image, axis=0)

        image -= mean
        image /= std

        cov = np.cov(image, rowvar=False)

        lambdas, p = np.linalg.eigh(cov)
        alphas = np.random.normal(0, 0.1, 3)

        delta = np.dot(p, alphas*lambdas)

        pca_augmentation_version_renorm_image = image + delta
        pca_color_image = pca_augmentation_version_renorm_image * std + mean
        pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype('uint8')

        return pca_color_image


def get_transforms():
    """Transformations for train and test images"""

    train_transform = transforms.Compose(
        [
            PCA_normalize(),
            transforms.ToTensor(),
        ]
    )
    return train_transform


def get_dataset(path):
    """Image dataset from path"""

    train_transform = get_transforms()
    train_dataset = datasets.ImageFolder(
        os.path.join(path, "train"), transform=train_transform
    )
    return train_dataset


def get_dataloaders(path, batch_size=128):
    """Image dataset loaders"""

    train_dataset = get_dataset(path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader



if __name__=='__main__':
    for k_class in glob(os.path.join('train','*')):
        path = os.path.join('train',k_class)
        expand_training_data(path)