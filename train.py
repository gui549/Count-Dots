import torch
import numpy as np

import argparse
import random
from torch.utils import data
from tqdm import tqdm
import wandb

from dataloader import DotsDataset
from utils import get_scheduler
from models import *  # Put model name here
from torchvision import transforms
from test import eval

import pdb


def train(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = args.epochs
    save_file_name = args.save_path

    if args.data_mode == 'dots':
        root_dir = './datasets/Dots/'
        test_dir = './datasets/TestDots/'
        trainset = DotsDataset(root_dir, transforms.Compose([transforms.RandomHorizontalFlip(),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize([0.9703, 0.9705, 0.9701], [0.1376, 0.1375, 0.1382])]))
        testset = DotsDataset(test_dir, transforms.Compose([transforms.ToTensor(), 
                                                            transforms.Normalize([0.9703, 0.9705, 0.9701], [0.1376, 0.1375, 0.1382])]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError

    if args.log:
        wandb.init(project='project name', config = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'save_file_name': args.save_path
        })
    
    if args.model == 'resnet':
        if args.load_path:
            model = torch.load(args.load_path)
        else:
            model = resnet18(args.num_classes)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == 'resnet_scalar':
        if args.load_path:
            model = torch.load(args.load_path)
        else:
            model = resnet18_scalar()
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters() ,lr=args.lr, betas=(0.9, 0.999))
    lr_scheduler = get_scheduler(optimizer, args)
    
    for epoch in range(epochs):
        running_loss = 0.
        
        for b, batch in tqdm(enumerate(train_loader), ncols=80, desc='Epoch {}'.format(epoch), total=len(train_loader)):
            images = batch['image'].to(device)
            if args.model == 'resnet_scalar':
                labels = batch['label'].to(device, torch.float).view(-1, 1) # set dytpe due to MSE loss
            else:
                labels = batch['label'].to(device)
                
            model.train()
            optimizer.zero_grad()

            preds = model(images)
            loss = torch.sum(criterion(preds, labels))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        
        if args.log:
            wandb.log({'running loss':running_loss/len(train_loader)}, commit=False)

        eval(model, test_loader, args)
        
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model, './experiments/'+save_file_name+'_{}.pth'.format(epoch))
    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train dot counter')
    parser.add_argument("-e", "--epochs", help="training epochs", default=120, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=5e-4, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [davis, ]", default='dots', type=str)
    parser.add_argument("-n", "--num_classes", help="number of classes", default='31', type=int)
    parser.add_argument("-m", "--model", help="use which model, [resnet, resnet_scalar, ]", default='resnet', type=str)
    parser.add_argument("-s", "--scheduler", help = "step, plateau, cosine, lambda", default = 'step', type =str)
    parser.add_argument("-f", "--save_path", help='save path for model', default = 'base', type=str)
    parser.add_argument("--load_path", help='load path for model', default = None, type=str)
    parser.add_argument("-l", "--log", help='log to wandb', action='store_true')
    args = parser.parse_args()

    random_seed = 4885
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    train(args)
