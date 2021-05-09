import torch
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, args):
    if args.scheduler == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - args.num_epoch // 2) / float(args.num_epoch // 2 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler
