import math

import numpy as np
import torch


class AverageMeter:
    """Computes and stores the running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def adjust_learning_rate(optimizer, epoch: int, args):
    """
    Adjusts the optimizer learning rate according to the schedule specified
    in args.lradj.

    Schedules:
      'type1'   — aggressive halving every epoch (legacy, not recommended)
      'type2'   — stepped milestones
      'constant' — no adjustment
      'cosine'  — handled externally by CosineAnnealingWarmRestarts in the
                  experiment runner; calling this function is a no-op.
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8,
        }
    elif args.lradj == 'constant':
        lr_adjust = {}
    elif args.lradj == 'cosine':
        # Cosine schedule is managed by the scheduler in exp_long_term_forecasting.py
        return
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr:.2e}')


class EarlyStopping:
    """
    Early stopping monitor.

    Saves model checkpoint whenever validation loss improves; stops training
    if it has not improved for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        """
        Args:
            patience: How many epochs to wait after last improvement.
            verbose:  Print a message when validation loss improves.
            delta:    Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, path: str):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). '
                f'Saving model ...'
            )
        torch.save(model.state_dict(), path + '/checkpoint.pth')
        self.val_loss_min = val_loss
