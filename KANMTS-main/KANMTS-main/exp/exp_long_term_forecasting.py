import random
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Helpers for GPU memory measurement
# ---------------------------------------------------------------------------

def _reset_gpu_stats(device):
    """Reset peak memory stats on the given CUDA device."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _get_gpu_allocated_mb(device):
    """Return currently allocated GPU memory in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return 0.0


def _get_gpu_peak_mb(device):
    """Return peak GPU memory allocated (since last reset) in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def _get_gpu_reserved_mb(device):
    """Return total GPU memory reserved by the caching allocator in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return torch.cuda.memory_reserved(device) / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    # ------------------------------------------------------------------
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # ------------------------------------------------------------------
    # Utility: forward pass (shared between vali / test)
    # ------------------------------------------------------------------
    def _forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
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
        return outputs

    # ------------------------------------------------------------------
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))

        self.model.train()
        return total_loss.avg

    # ------------------------------------------------------------------
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        result_path = './result/results.txt'
        os.makedirs('./result', exist_ok=True)

        # Reproducibility
        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.set_num_threads(6)

        train_steps  = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim  = self._select_optimizer()
        criterion    = self._select_criterion()
        scaler       = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # ---- GPU memory: record baseline before training starts --------
        _reset_gpu_stats(self.device)
        train_start_mem_mb  = _get_gpu_allocated_mb(self.device)
        train_start_time    = time.time()
        time_now            = train_start_time
        # ----------------------------------------------------------------

        for epoch in range(self.args.train_epochs):
            iter_count  = 0
            train_loss  = []
            self.model.train()
            epoch_time  = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                f_dim   = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss    = criterion(outputs, batch_y)

                if (i + 1) % 100 == 0:
                    loss_val = loss.item()
                    train_loss.append(loss_val)
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\titers: {i+1}, epoch: {epoch+1} | loss: {loss_val:.7f}')
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now   = time.time()

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_cost = time.time() - epoch_time
            print(f'Epoch: {epoch+1} cost time: {epoch_cost:.2f}s')

            train_loss = np.average(train_loss) if train_loss else float('nan')
            vali_loss  = self.vali(vali_data, vali_loader, criterion)
            test_loss  = self.vali(test_data,  test_loader,  criterion)

            print(f'Epoch: {epoch+1}, Steps: {train_steps} | '
                  f'Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}')

            with open(result_path, 'a', encoding='utf-8') as f:
                f.write(f'Epoch: {epoch+1}, Steps: {train_steps}, '
                        f'Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}, '
                        f'Test Loss: {test_loss:.7f}, Time Cost: {epoch_cost:.4f}s\n')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # ---- GPU memory: measure after training ends -------------------
        #   synchronize so all CUDA work is complete before we measure
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        train_total_time_s   = time.time() - train_start_time
        train_end_mem_mb     = _get_gpu_allocated_mb(self.device)
        train_peak_mem_mb    = _get_gpu_peak_mb(self.device)       # peak since reset at start
        train_reserved_mb    = _get_gpu_reserved_mb(self.device)   # total reserved by allocator
        train_delta_mem_mb   = train_end_mem_mb - train_start_mem_mb

        print(f'\n=== Training Memory & Time ===')
        print(f'Total Training Time        : {train_total_time_s:.2f}s')
        print(f'GPU Allocated (start)      : {train_start_mem_mb:.2f} MB')
        print(f'GPU Allocated (end)        : {train_end_mem_mb:.2f} MB')
        print(f'GPU Allocated delta        : {train_delta_mem_mb:.2f} MB')
        print(f'GPU Peak Allocated         : {train_peak_mem_mb:.2f} MB  <-- most meaningful')
        print(f'GPU Reserved by allocator  : {train_reserved_mb:.2f} MB')

        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'\n=== Training Memory & Time ===\n')
            f.write(f'Total Training Time        : {train_total_time_s:.2f}s\n')
            f.write(f'GPU Allocated (start)      : {train_start_mem_mb:.2f} MB\n')
            f.write(f'GPU Allocated (end)        : {train_end_mem_mb:.2f} MB\n')
            f.write(f'GPU Allocated delta        : {train_delta_mem_mb:.2f} MB\n')
            f.write(f'GPU Peak Allocated         : {train_peak_mem_mb:.2f} MB\n')
            f.write(f'GPU Reserved by allocator  : {train_reserved_mb:.2f} MB\n')
        # ----------------------------------------------------------------

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)

        return self.model

    # ------------------------------------------------------------------
    def test(self, setting, test=0):
        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.set_num_threads(6)

        result_path = './result/results.txt'
        os.makedirs('./result', exist_ok=True)

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()

        # ---- GPU memory: reset before test loop ------------------------
        _reset_gpu_stats(self.device)
        # Track per-batch inference time separately (wall-clock)
        total_inference_time_s = 0.0
        # ----------------------------------------------------------------

        test_start_time   = time.time()
        test_start_mem_mb = _get_gpu_allocated_mb(self.device)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # ---- time only the forward pass ------------------------
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                batch_start = time.perf_counter()

                outputs = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)   # wait for GPU to finish
                total_inference_time_s += time.perf_counter() - batch_start
                # --------------------------------------------------------

                f_dim   = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))

        # ---- GPU memory: read after all batches done -------------------
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        test_total_time_s  = time.time() - test_start_time
        test_end_mem_mb    = _get_gpu_allocated_mb(self.device)
        test_peak_mem_mb   = _get_gpu_peak_mb(self.device)      # peak since reset before loop
        test_reserved_mb   = _get_gpu_reserved_mb(self.device)
        test_delta_mem_mb  = test_end_mem_mb - test_start_mem_mb
        # ----------------------------------------------------------------

        mse_val = mse.avg
        mae_val = mae.avg

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f'\n=== Test Results ===')
        print(f'MSE: {mse_val:.6f}  MAE: {mae_val:.6f}')
        print(f'\n=== Test Memory & Time ===')
        print(f'Total test wall-clock time : {test_total_time_s:.2f}s')
        print(f'Pure inference time        : {total_inference_time_s:.2f}s  (forward pass only)')
        print(f'GPU Allocated (start)      : {test_start_mem_mb:.2f} MB')
        print(f'GPU Allocated (end)        : {test_end_mem_mb:.2f} MB')
        print(f'GPU Allocated delta        : {test_delta_mem_mb:.2f} MB')
        print(f'GPU Peak Allocated         : {test_peak_mem_mb:.2f} MB  <-- most meaningful')
        print(f'GPU Reserved by allocator  : {test_reserved_mb:.2f} MB')
        print(f'Trainable parameters       : {total_params:,}')

        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(f'\n=== Test Results ===\n')
            f.write(f'MSE: {mse_val:.6f}  MAE: {mae_val:.6f}\n')
            f.write(f'\n=== Test Memory & Time ===\n')
            f.write(f'Total test wall-clock time : {test_total_time_s:.2f}s\n')
            f.write(f'Pure inference time        : {total_inference_time_s:.2f}s\n')
            f.write(f'GPU Allocated (start)      : {test_start_mem_mb:.2f} MB\n')
            f.write(f'GPU Allocated (end)        : {test_end_mem_mb:.2f} MB\n')
            f.write(f'GPU Allocated delta        : {test_delta_mem_mb:.2f} MB\n')
            f.write(f'GPU Peak Allocated         : {test_peak_mem_mb:.2f} MB\n')
            f.write(f'GPU Reserved by allocator  : {test_reserved_mb:.2f} MB\n')
            f.write(f'Trainable parameters       : {total_params}\n')

        return
