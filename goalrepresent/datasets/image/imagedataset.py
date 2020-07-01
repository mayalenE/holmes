import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomHorizontalFlip, RandomVerticalFlip

import goalrepresent as gr
from goalrepresent.datasets.image.preprocess import RandomCenterCrop, RandomRoll, RandomSphericalRotation

to_tensor = ToTensor()
to_PIL_image = ToPILImage()

# ===========================
# get dataset function
# ===========================

def get_dataset(dataset_name):
    """
    dataset_name: string such that the model called is <dataset_name>Dataset
    """
    return eval("gr.datasets.{}Dataset".format(dataset_name.upper()))

# ===========================
# Mixed Datasets
# ===========================

class MIXEDDataset(Dataset):

    @staticmethod
    def default_config():
        default_config = gr.Config()
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None
        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        # initially dataset lists are empty
        self.n_images = 0
        self.images = torch.FloatTensor([]) # list or torch tensor of size N*C*H*W
        self.labels = torch.LongTensor([]) # list or torch tensor
        self.datasets_ids = [] # list of the dataset idx each image is coming from

        self.datasets = {}
        for dataset in self.config.datasets:
            dataset_class = get_dataset(dataset["name"])
            self.datasets[dataset["name"]] = dataset_class(config=dataset.config)

        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def update(self, n_images, images, labels=None, datasets_ids=None):
        """ Update the current dataset lists """
        if labels is None:
            labels = torch.LongTensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print(
            'ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(
                n_images, images.shape[0], labels.shape[0]))

        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

        if datasets_ids is not None:
            self.datasets_ids = datasets_ids

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            datasets_names = list(self.datasets.keys())
            if len(datasets_names) == 1:
                img_tensor = self.datasets[datasets_names[0]].augment(img_tensor)
            elif len(self.datasets_ids) > 0:
                dataset_id = self.datasets_ids[idx]
                if dataset_id is not None:  # if generated data (None) we do not augment
                    img_tensor = self.datasets[dataset_id].augment(img_tensor)
            else:
                raise ValueError("Cannot augment data if dataset_ids is not given")

        if self.transform is not None:
            datasets_names = list(self.datasets.keys())
            if len(datasets_names) == 1:
                img_tensor = self.datasets[datasets_names[0]].transform(img_tensor)
            elif len(self.datasets_ids) > 0:
                dataset_id = self.datasets_ids[idx]
                if dataset_id is not None:
                    img_tensor = self.datasets[dataset_id].transform(img_tensor)
            else:
                raise ValueError("Cannot augment data if dataset_ids is not given")

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            datasets_names = list(self.datasets.keys())
            if len(datasets_names) == 1:
                label = self.datasets[datasets_names[0]].target_transform(label)
            elif len(self.datasets_ids) > 0:
                dataset_id = self.datasets_ids[idx]
                if dataset_id is not None:
                    label = self.datasets[dataset_id].target_transform(label)
            else:
                raise ValueError("Cannot augment data if dataset_ids is not given")

        return {'obs': img_tensor, 'label': label, 'index': idx}


    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))

# ===========================
# Lenia Dataset
# ===========================

class LENIADataset(Dataset):
    """ Lenia dataset"""

    @staticmethod
    def default_config():
        default_config = gr.Config()

        # load data
        default_config.data_root = None
        default_config.split = "train"

        # process data
        default_config.preprocess = None
        default_config.img_size = None
        default_config.data_augmentation = False
        default_config.transform = None
        default_config.target_transform = None

        return default_config

    def __init__(self, config=None, **kwargs):
        self.config = gr.config.update_config(kwargs, config, self.__class__.default_config())

        if self.config.data_root is None:
            self.n_images = 0
            self.images = torch.zeros((0, 1, self.config.img_size[0], self.config.img_size[0]))
            if self.config.img_size is not None:
                self.img_size = self.config.img_size
                self.n_channels = 1
            self.labels = torch.zeros((0, 1), dtype=torch.long)

        else:
            # load HDF5 lenia dataset
            dataset_filepath = os.path.join(self.config.data_root, 'dataset', 'dataset.h5')
            with h5py.File(dataset_filepath, 'r') as file:
                if 'n_data' in file[self.config.split]:
                    self.n_images = int(file[self.config.split]['n_data'])
                else:
                    self.n_images = int(file[self.config.split]['observations'].shape[0])

                self.has_labels = bool('labels' in file[self.config.split])
                if self.has_labels:
                    self.labels = torch.LongTensor(file[self.config.split]['labels'])
                else:
                    self.labels = torch.LongTensor([-1] * self.n_images)

                self.images = torch.Tensor(file[self.config.split]['observations']).float()
                if self.config.preprocess is not None:
                    self.images = self.config.preprocess(self.images)

                self.n_channels = 1
                if self.images.ndim == 3:
                    self.images = self.images.unsqueeze(1)
                self.img_size = (self.images.shape[2], self.images.shape[3])


        # data augmentation boolean
        self.data_augmentation = self.config.data_augmentation
        if self.data_augmentation:
            # LENIA Augment
            self.random_center_crop = RandomCenterCrop(p=0.6, crop_ratio=(1., 2.), keep_img_size=True)
            self.random_roll = RandomRoll(p_x=0.6, p_y=0.6, max_dx=0.5, max_dy=0.5, img_size=self.img_size)
            self.random_spherical_rotation = RandomSphericalRotation(p=0.6, max_degrees=20, n_channels=self.n_channels, img_size=self.img_size)
            self.random_horizontal_flip = RandomHorizontalFlip(0.2)
            self.random_vertical_flip = RandomVerticalFlip(0.2)
            self.augment = Compose([self.random_center_crop, self.random_roll, self.random_spherical_rotation, to_PIL_image, self.random_horizontal_flip, self.random_vertical_flip, to_tensor])


        # the user can additionally specify a transform in the config
        self.transform = self.config.transform
        self.target_transform = self.config.target_transform

    def update(self, n_images, images, labels=None):
        """update online the dataset"""
        if labels is None:
            labels = torch.Tensor([-1] * n_images)
        assert n_images == images.shape[0] == labels.shape[0], print(
            'ERROR: the given dataset size ({0}) mismatch with observations size ({1}) and labels size ({2})'.format(
                n_images, images.shape[0], labels.shape[0]))

        self.n_images = int(n_images)
        self.images = images
        self.labels = labels

    def get_image(self, image_idx, augment=False, transform=True):
        image = self.images[image_idx]
        if augment and self.data_augmentation:
            image = self.augment(image)
        if transform and self.transform is not None:
            image = self.transform(image)
        return image

    def get_augmented_batch(self, image_ids, augment=True, transform=True):
        images_aug = []
        for img_idx in image_ids:
            image_aug = self.get_image(img_idx, augment=augment, transform=transform)
            images_aug.append(image_aug)
        images_aug = torch.stack(images_aug, dim=0)
        return images_aug

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # image
        img_tensor = self.images[idx]

        if self.data_augmentation:
            img_tensor = self.augment(img_tensor)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # label
        if self.labels[idx] is not None and not np.isnan(self.labels[idx]):
            label = int(self.labels[idx])
        else:
            label = -1

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {'obs': img_tensor, 'label': label, 'index': idx}

    def save(self, output_npz_filepath):
        np.savez(output_npz_filepath, n_images=self.n_images, images=np.stack(self.images),
                 labels=np.asarray(self.labels))
        return