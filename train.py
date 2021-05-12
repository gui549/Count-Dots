import torch
import numpy as np

import argparse
import random
from tqdm import tqdm
import wandb

from dataloader import DotsDataset
from utils import get_scheduler
from models import * # Put model name here
from torchvision import transforms
from test import Eval

import pdb


def train(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = args.epochs
    save_file_name = args.save_path

    if args.data_mode == 'dots':
        root_dir = './datasets/Dots/'
        test_dir = './datasets/TestDots/'
        dataset = DotsDataset(root_dir, transforms.Compose([transforms.ToTensor()]))
        testset = DotsDataset(test_dir, transforms.Compose([transforms.ToTensor()]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError

    if args.log:
        wandb.init(project='project name', config = {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'save_file_name': args.save_path
        })
    
    if args.model == 'resnet':
        model = resnet18(args.num_classes)
    elif args.model == '':
        pass

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters() ,lr=args.lr, betas=(0.9, 0.999))
    lr_scheduler = get_scheduler(optimizer, args)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.
        
        for b, batch in tqdm(enumerate(loader), ncols=80, desc='Epoch {}'.format(epoch), total=len(loader)):
            images = batch['image'].to(device)
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
            wandb.log({'running loss':running_loss/len(loader)}, commit=False)

        Eval(model, test_loader, args)
        
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
    parser.add_argument("-m", "--model", help="use which model, [resnet, ]", default='resnet', type=str)
    parser.add_argument("-s", "--scheduler", help = "step, plateau, cosine, lambda", default = 'step', type =str)
    parser.add_argument("-f", "--save_path", help='save path for model', default = 'base', type=str)
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
