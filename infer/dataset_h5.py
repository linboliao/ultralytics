import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, file_path, wsi, pretrained=False, custom_transforms=None, custom_downsample=1, target_patch_size=-1,
                 fast_read=False,
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        self.roi_transforms = custom_transforms

        self.file_path = file_path

        self.fast_read = fast_read
        if fast_read:
            print('Loading coord file:', self.file_path)
            with h5py.File(self.file_path, 'r') as hdf5_file:
                self.coords = hdf5_file['coords'][:]
            print('coords num:', len(self.coords))

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None

        # self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        if self.fast_read:
            coord = self.coords[idx]
        else:
            # supoort old style
            with h5py.File(self.file_path, 'r') as hdf5_file:
                coord = hdf5_file['coords'][idx]

        try:
            img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        except Exception as e:
            print('Failed to read region: {},{}'.format(*coord))
            print('Exception: {}'.format(e))
            img = np.ones(shape=(self.patch_size, self.patch_size, 3)).astype('uint8') * 255
            img = Image.fromarray(img)

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img)
        return img, coord


class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, dtype={'case_id': str, 'slide_id': str})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
