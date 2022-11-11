import time
from pathlib import Path
from random import randint, random
import torch as th
from torch.utils.data import Dataset
import numpy as np


def get_image_files_dict(base_path):
    image_files = [
        *base_path.glob("**/*.npy"),
    ]
    return {image_file.stem: image_file for image_file in image_files}


def get_para_files_dict(base_path):
    text_files = [*base_path.glob("**/*.npy")]
    return {text_file.stem: text_file for text_file in text_files}


def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)


class TextImageDataset(Dataset):
    def __init__(
        self,
        folder="",
        image_size=64,
        shuffle=False,
        uncond_p=0.0,
        n_param=2,
        drop_para=False
    ):
        super().__init__()
        folder = Path(folder)

        self.image_files = get_image_files_dict(folder)
        self.text_files = get_para_files_dict(folder)
        self.keys = get_shared_stems(self.image_files, self.text_files)
        print(f"Found {len(self.keys)} images.")
        print(f"Using {len(self.text_files)} parameter files.")

        self.n_param = n_param

        self.shuffle = shuffle
        self.prefix = folder
        self.image_size = image_size
        self.uncond_p = uncond_p
        self.drop_para=drop_para

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_para(self, ind):
        key = self.keys[ind]
        paramfilename = str(self.text_files[key])
        paramcache=np.array([float(paramfilename.split('/')[-1].split('_')[2]),np.log10(float(paramfilename.split('/')[-1].split('_')[1]))]) #image file names: *_param2_param1*.npy
        paramcache=(paramcache-np.array([4.,1.]))/np.array([2.,1.398]).astype(np.float32) #data normalized to [0,1], original parameter range: [4,6] and [1,2.398]
        return paramcache

    def __getitem__(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]
        if random() < self.uncond_p and self.drop_para == True:
            tokens = np.float32(np.array([0]).repeat(self.n_param)) #array([0,0]) for uncondional training
        else:
            tokens = self.get_para(ind)

        try:
            original_image = np.float32(np.load(image_file))
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        arr = np.expand_dims(original_image,axis=0) # only one channel
        arr = arr*2 - 1 # image array already normalized to [0,1], here further to [-1,1]
        return th.tensor(arr),th.tensor(np.float32(tokens))

