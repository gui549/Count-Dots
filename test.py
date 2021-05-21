import torch
import numpy as np

import argparse
import random
import wandb

from dataloader import DotsDataset
from torchvision import transforms

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
   
def get_num_correct(top_pred_ids, labels):
    return top_pred_ids.eq(labels).sum().item()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test dot counter')
    parser.add_argument("-n", "--num_classes", help="number of classes", default='31', type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [davis, ]", default='dots', type=str)
    parser.add_argument("-f", "--load_path", help="use which saved model", default=None, type=str)
    parser.add_argument("-l", "--log", help='log to wandb', action='store_true')
    args = parser.parse_args()

    if args.load_path == None:
        raise SyntaxError("Enter a valid load path")

    if args.log:
        wandb.init(project='project name')

    model = torch.load(args.load_path)

    if args.data_mode == 'dots':
        test_dir = './datasets/TestDots/'
        testset = DotsDataset(test_dir, transforms.Compose([transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    eval(model, test_loader, args)

    wandb.finish()