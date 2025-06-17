import os, glob
import torch, sys
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class JHUBrainDataset(Dataset):
    def __init__(self, data_path1, data_path2, transforms):
        self.paths1 = data_path1
        self.paths2 = data_path2
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path1 = self.paths1[index]
        path2 = self.paths2[index]

        image_nii1 = nib.load(path1)
        image_nii2 = nib.load(path2)


        x = image_nii1.get_fdata()
        y = image_nii2.get_fdata()

        x, y = self.transforms([x[None, ...], y[None, ...]])

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths1)




class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path1, data_path2, transforms):
        self.paths1 = data_path1
        self.paths2 = data_path2
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path1 = self.paths1[index]
        path2 = self.paths2[index]
        x = nib.load(path1)
        y = nib.load(path2)
        x = x.get_fdata()
        y = y.get_fdata()


        seg_dir1 = '/data/T2-T2_multi/T2time2 label/'
        seg_dir2 = '/data/T2-T2_multi/T2time1 label/'
        x_seg_path = os.path.join(seg_dir1, os.path.basename(path1))
        y_seg_path = os.path.join(seg_dir2, os.path.basename(path2))
        x_seg = nib.load(x_seg_path)
        y_seg = nib.load(y_seg_path)
        x_seg = x_seg.get_fdata()
        y_seg = y_seg.get_fdata()

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths1)


