import torch
import torch.utils.data.Dataset as Dataset
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
    def __init__(self, root_dir, train_transforms):

        self.transform = train_transforms

        self.total_list = os.listdir(root_dir)

        self.total_imgs = len(total_list)
        self.total_list = total_list
        
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1] # {label}_filenum.jpg
        label = file_name.split('_')[0]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        batch = {
            'image': img,
            'label': label
        }

        return batch