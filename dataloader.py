import torch
import argparse
import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

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
        self.class_num = len(set([i.split('_')[0] for i in self.total_list]))
        
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1] # {label}_filenum.jpg
        label = int(file_name.split('_')[0])
        img = Image.open(os.path.join(self.root_dir,img_path))

        if self.transform is not None:
            img = self.transform(img)

        batch = {
            'image': img,
            'label': label
        }

        return batch

def get_mean_std(dataset):
    imgs = torch.stack([a['image'] for a in dataset], dim=3)
    temp = imgs.view(3, -1)
    mean_ = temp.mean(dim=1)
    std_ = temp.std(dim=1)
    return mean_, std_

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate mean, std')
    parser.add_argument("-f", "--data_path", help="data path", default="./datasets/Dots/", type=str)
    args = parser.parse_args()

    trainset = DotsDataset(args.data_path, transforms.Compose([transforms.ToTensor()]))
    mean_, std_ = get_mean_std(trainset)
    print(args.data_path)
    print("mean : ", mean_, "std :", std_)
