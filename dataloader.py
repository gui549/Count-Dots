import torch
from torch.utils.data import Dataset
from PIL import Image

import os

class DotsDataset(Dataset):
    '''
    Dataloader for 
    directory structure
        root/{label}_filenum.jpg

    Store image directories at init phase

    Returns image, label
    '''
    def __init__(self, root_dir, train_transforms=None):

        self.transform = train_transforms

        self.total_list = os.listdir(root_dir)
        self.root_dir = root_dir
        self.total_imgs = len(self.total_list)
        self.total_list = self.total_list
        
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1] # {label}_filenum.jpg
        label = int(file_name.split('_')[0])
        img = Image.open(self.root_dir + img_path)

        if self.transform is not None:
            img = self.transform(img)

        batch = {
            'image': img,
            'label': label
        }

        return batch