import random
import math

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
from layers.KANLinear import KANLinear

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import psutil


def combined_loss(pred: torch.Tensor, true: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Combined MSE + MAE loss (kept for backward compatibility).
    Default training now uses Huber loss instead.
    """
    mse = nn.functional.mse_loss(pred, true)
    mae = nn.functional.l1_loss(pred, true)
    return alpha * mse + (1.0 - alpha) * mae


def huber_loss(pred: torch.Tensor, true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss (Smooth L1) with configurable delta.

    Behaves like MSE for |error| < delta and like MAE for |error| >= delta.
    - More robust to outliers than pure MSE
    - Differentiable everywhere (unlike MAE)
    - Directly related to MAE near zero
    Best single-objective tradeoff for time-series forecasting.

    Args:
        pred:  Model predictions.
        true:  Ground-truth targets.
        delta: Threshold where loss transitions from L2 to L1. Default 1.0.
    Returns:
        Scalar loss tensor.
    """
    return nn.functional.huber_loss(pred, true, delta=delta)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        AdamW with weight decay provides built-in L2 regularization,
        which is more principled than plain Adam for networks with KAN
        spline weights that can grow large.
        """
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 1e-4),
            betas=(0.9, 0.95),
        )
        return model_optim

    def _select_criterion(self):
        """
        Loss selection via args.loss:
          'huber'    (default) — Huber/SmoothL1, best outlier-robust tradeoff
          'combined' — 0.7*MSE + 0.3*MAE
          'MSE'      — pure MSE
        """
        loss_type = getattr(self.args, 'loss', 'huber')
        if loss_type == 'MSE':
            return nn.MSELoss()
        if loss_type == 'combined':
            return combined_loss
        return huber_loss   # default: Huber

    # ─────────────────────────────────────────────────────────────────────────

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # For validation we always use MSE to have a stable comparable metric
                loss = nn.functional.mse_loss(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))

        self.model.train()
        return total_loss.avg

    # ─────────────────────────────────────────────────────────────────────────

    def _update_kan_grids(self, train_loader):
        """
        Update KAN spline grids using a single pass of training data.
        Call this once at the start of training so grids adapt to real
        data distributions before optimisation begins.
        """
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i >= 3:   # use first 3 batches only
                    break
                batch_x = batch_x.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :].to(self.device), dec_inp], dim=1)
                # Forward pass — intermediary acts are cached inside each KANLinear
                _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Now update grids for all KANLinear layers that have cached activations
            for module in self.model.modules():
                if isinstance(module, KANLinear) and hasattr(module, 'acts') and module.acts is not None:
                    acts = module.acts
                    if acts.dim() == 3:
                        acts = acts.reshape(-1, acts.size(-1))
                    try:
                        module.update_grid(acts)
                    except Exception:
                        pass   # skip if shapes mismatch

        self.model.train()
        print("KAN grids updated from training data distribution.")

    # ─────────────────────────────────────────────────────────────────────────

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # ── Reproducibility ───────────────────────────────────────────────────
        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.set_num_threads(6)

        time_now = time.time()
        train_steps = len(train_loader)
        patience = getattr(self.args, 'patience', 7)
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # ── Cosine Annealing LR Scheduler ─────────────────────────────────────
        # Restarts every T_0=5 epochs; smoother than the original type1 halving.
        use_cosine = getattr(self.args, 'lradj', 'cosine') == 'cosine'
        scheduler = None
        if use_cosine:
            scheduler = CosineAnnealingWarmRestarts(
                model_optim,
                T_0=max(5, self.args.train_epochs // 4),
                T_mult=1,
                eta_min=1e-7,
            )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ── Adaptive KAN Grid Initialisation ─────────────────────────────────
        print("Initialising KAN grids from training data...")
        self._update_kan_grids(train_loader)

        # ── Logging helpers ───────────────────────────────────────────────────
        def save_epoch_result(epoch, train_loss, vali_loss, test_loss, time_cost):
            with open('./result/results.txt', 'a', encoding='utf-8', errors='replace') as f:
                f.write(
                    f"Epoch: {epoch+1}, Steps: {train_steps}, "
                    f"Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, "
                    f"Test Loss: {test_loss:.7f}, Time Cost: {time_cost:.4f}s\n"
                )

        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss

        # ── Training Loop ─────────────────────────────────────────────────────
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_list = []   # BUG FIX: collect ALL batch losses, not just every-100

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = outputs[0]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_trimmed = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y_trimmed)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_trimmed = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y_trimmed)

                # BUG FIX: record every batch loss
                train_loss_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                # ── Backward pass ─────────────────────────────────────────────
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # Gradient clipping to prevent exploding KAN spline gradients
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")

            # BUG FIX: train_loss_list is always populated now
            train_loss = np.mean(train_loss_list) if train_loss_list else float('nan')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            time_cost = time.time() - epoch_time
            save_epoch_result(epoch, train_loss, vali_loss, test_loss, time_cost)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # ── LR update ─────────────────────────────────────────────────────
            if scheduler is not None:
                scheduler.step()
                current_lr = model_optim.param_groups[0]['lr']
                print(f"LR updated to {current_lr:.2e}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # ── After training ────────────────────────────────────────────────────
        end_time = time.time()
        end_memory = process.memory_info().rss
        total_time = end_time - start_time
        total_memory = (end_memory - start_memory) / 1024  # bytes → KB

        print(f"Total Training Time: {total_time:.2f}s")
        print(f"Total Memory Usage:  {total_memory:.2f} KB")

        with open('./result/results.txt', 'a', encoding='utf-8') as f:
            f.write(f"Total Training Time: {total_time:.2f} seconds\n")
            f.write(f"Total Training Memory Usage: {total_memory:.2f} KB\n")

        # Load best checkpoint
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model

    # ─────────────────────────────────────────────────────────────────────────

    def test(self, setting, test=0):
        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.set_num_threads(6)

        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('Loading model checkpoint...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        mse_metric = nn.MSELoss()
        mae_metric = nn.L1Loss()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()

        predictions = []
        ground_truth = []
        total_inference_time = 0.0
        total_memory_used = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_start_time = time.time()
                proc = psutil.Process(os.getpid())
                batch_start_memory = proc.memory_info().rss / (1024 * 1024)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                predictions.append(outputs.cpu().numpy())
                ground_truth.append(batch_y.cpu().numpy())

                batch_end_time = time.time()
                batch_end_memory = proc.memory_info().rss / (1024 * 1024)
                total_inference_time += batch_end_time - batch_start_time
                total_memory_used += batch_end_memory - batch_start_memory

                mse_meter.update(mse_metric(outputs, batch_y).item(), batch_x.size(0))
                mae_meter.update(mae_metric(outputs, batch_y).item(), batch_x.size(0))

        print(f"Total Inference Time: {total_inference_time:.2f}s")
        print(f"Total Inference Memory: {total_memory_used:.2f} MB")

        mse = mse_meter.avg
        mae = mae_meter.avg
        print(f"MSE: {mse:.6f}  |  MAE: {mae:.6f}")

        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)

        with open('./result/results.txt', 'a', encoding='utf-8', errors='replace') as file:
            file.write(f'Total Inference Time: {total_inference_time:.2f}s\n')
            file.write(f'Total Inference Memory: {total_memory_used:.2f} MB\n')
            file.write(f'MSE: {mse}\n')
            file.write(f'MAE: {mae}\n')
            file.write(f'Total test time: {end_time - start_time:.2f} seconds\n')
            file.write(f'Test memory used: {end_memory - start_memory:.2f} MB\n')

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        with open('./result/results.txt', 'a') as f:
            f.write(f'Total trainable parameters: {total_params:,}\n')

        return
