import torch
import numpy as np

import argparse
import random
import wandb

from dataloader import DotsDataset
from torchvision import transforms
from models import *

import pdb

def eval(model, loader, args):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        model = model.to(device)
        model.eval()

        preds_correct = 0
        all_preds = torch.tensor([], dtype=torch.float32).to(device)
        all_labels = torch.tensor([], dtype=torch.float32).to(device)

        for batch in loader:
            images = batch['image'].to(device)
            
            if args.model == 'resnet':
                labels = batch['label'].to(device)
                preds = model(images) 
                refined_preds = preds.argmax(dim=1)
            elif args.model == 'resnet_scalar':
                labels = batch['label'].to(device).view(-1, 1)
                preds = model(images) 
                refined_preds = torch.round(preds)
            else :
                raise NotImplementedError

            preds_correct += get_num_correct(refined_preds, labels)
            all_preds = torch.cat((all_preds, refined_preds), 0)
            all_labels = torch.cat((all_labels, labels), 0)
        
        all_preds = (all_preds.cpu()).numpy()
        all_labels = (all_labels.cpu()).numpy()

        if args.log:
            if args.model == 'resnet':
                wandb.log({
                    "accuracy" : preds_correct / len(loader.dataset),
                    "conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds, class_names=[i for i in range(args.num_classes)])
                })
            elif args.model == 'resnet_scalar' :
                wandb.log({
                    "accuracy" : preds_correct / len(loader.dataset),
                })
        else:
            print("accuracy", preds_correct / len(loader.dataset))
   
def get_num_correct(preds, labels):
    return preds.eq(labels).sum().item()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test dot counter')
    parser.add_argument("-m", "--model", help="use which model, [resnet, resnet_scalar, ]", default='resnet', type=str)
    parser.add_argument("-f", "--load_path", help="use which saved model", default=None, type=str)
    parser.add_argument("-t", "--test_path", help="use which test set", default= './datasets/TestDots/', type=str)
    parser.add_argument("-l", "--log", help='log to wandb', action='store_true')

    args = parser.parse_args()

    if args.log:
        wandb.init(entity='kongjoo',
        project='Counting Stars', config = {
            'test_model': args.load_path,
            'test_set': args.test_path
        })

    if args.load_path == None:
        raise SyntaxError("Enter a valid load path")

    test_dir = args.test_path
    testset = DotsDataset(test_dir, transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.9703, 0.9705, 0.9701], [0.1376, 0.1375, 0.1382])]))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    if args.model == 'resnet':
        model = resnet18(testset.class_num)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == 'resnet_scalar':
        model = resnet18_scalar()
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    if args.load_path:
        model = torch.load(args.load_path)        

    eval(model, test_loader, args)
    wandb.finish()