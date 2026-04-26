import argparse
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.set_num_threads(6)

    parser = argparse.ArgumentParser(description='KANMTS — Improved Multivariate Time-Series Forecasting')

    # ── KAN-specific ──────────────────────────────────────────────────────────
    parser.add_argument('--grid_size', type=int, default=8,
                        help='KAN spline grid size (higher = more expressive). Was 4; raised to 8.')

    # ── Basic config ─────────────────────────────────────────────────────────
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status: 1=train, 0=test only')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='KANMTS',
                        help='model name, options: [KANMTS, SOFTS, ...]')

    # ── Data loader ───────────────────────────────────────────────────────────
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='[M] multivariate→multivariate, [S] univariate, [MS] multi→univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature for S/MS tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='time feature encoding freq: [s,t,h,d,b,w,m] or e.g. 15min')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')

    # ── Forecasting task ─────────────────────────────────────────────────────
    parser.add_argument('--seq_len', type=int, default=96, help='encoder input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='decoder start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input (variate) size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input (variate) size')
    parser.add_argument('--c_out', type=int, default=7, help='output (variate) size')
    parser.add_argument('--d_model', type=int, default=512, help='model embedding dimension')
    parser.add_argument('--d_core', type=int, default=128,
                        help='bottleneck dimension inside the KAN mixer (< d_model)')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--num_layers', type=int, default=2, help='number of channel MLP layers')
    parser.add_argument('--d_layers', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='feedforward network dimension')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='moving average window for series decomposition')
    parser.add_argument('--factor', type=int, default=1, help='attention factor')
    parser.add_argument('--distil', action='store_false',
                        help='disable distilling in encoder', default=True)

    # ── Regularization ────────────────────────────────────────────────────────
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate (was 0.0; raised to 0.1 for regularization)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay (L2 regularization)')

    # ── Attention ─────────────────────────────────────────────────────────────
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of heads in VariateAttention module')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_attention', action='store_true',
                        help='output attention weights from encoder')
    parser.add_argument('--attention_type', type=str, default='full', help='attention type')

    # ── Normalization ─────────────────────────────────────────────────────────
    parser.add_argument('--use_norm', type=int, default=1,
                        help='use RevIN normalization (1=True)')

    # ── Optimization ─────────────────────────────────────────────────────────
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num workers')
    parser.add_argument('--itr', type=int, default=1, help='number of experiment repetitions')
    parser.add_argument('--train_epochs', type=int, default=30,
                        help='training epochs (was 20; raised to 30 for more budget)')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--patience', type=int, default=7,
                        help='early stopping patience (was 3; raised to 7)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='optimizer initial LR (was 1e-4; increased for AdamW + cosine)')
    parser.add_argument('--des', type=str, default='test', help='experiment description')
    parser.add_argument('--loss', type=str, default='combined',
                        help='loss function: [combined (MSE+MAE), MSE]')
    parser.add_argument('--lradj', type=str, default='cosine',
                        help='LR schedule: [cosine, type1, type2, constant]')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training', default=False)

    # ── GPU ───────────────────────────────────────────────────────────────────
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU if available')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple GPUs', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='comma-separated GPU device ids')

    # ── Checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument('--save_model', action='store_true',
                        help='keep checkpoint after training')

    # ── Mixer dims ────────────────────────────────────────────────────────────
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='hidden dim inside Token/ChannelMixingKAN (was 20)')
    parser.add_argument('--n_layers', type=int, default=2)

    # ── Parse ─────────────────────────────────────────────────────────────────
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(x) for x in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    # ─────────────────────────────────────────────────────────────────────────

    def train(args=args):
        setting = (
            '{task}_{mid}_{model}_{data}_ft{feat}_sl{sl}_ll{ll}_pl{pl}'
            '_dm{dm}_dc{dc}_el{el}_dl{dl}_df{df}_fc{fc}_dt{dt}_{des}'
            '_nl{nl}_hd{hd}_bs{bs}_lr{lr}_te{te}_gs{gs}_nh{nh}'
        ).format(
            task=args.task_name,
            mid=args.model_id,
            model=args.model,
            data=args.data,
            feat=args.features,
            sl=args.seq_len,
            ll=args.label_len,
            pl=args.pred_len,
            dm=args.d_model,
            dc=args.d_core,
            el=args.e_layers,
            dl=args.d_layers,
            df=args.d_ff,
            fc=args.factor,
            dt=args.distil,
            des=args.des,
            nl=args.num_layers,
            hd=args.hidden_dim,
            bs=args.batch_size,
            lr=args.learning_rate,
            te=args.train_epochs,
            gs=args.grid_size,
            nh=args.n_heads,
        )

        exp = Exp(args)

        with open('./result/results.txt', 'a', encoding='utf-8', errors='replace') as f:
            f.write(f'setting: {setting}\n')

        print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)

        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────

    if args.is_training:
        train(args)
    else:
        setting = (
            '{task}_{mid}_{model}_{data}_ft{feat}_sl{sl}_ll{ll}_pl{pl}'
            '_dm{dm}_el{el}_dl{dl}_df{df}_fc{fc}_eb{eb}_dt{dt}_{des}'
        ).format(
            task=args.task_name,
            mid=args.model_id,
            model=args.model,
            data=args.data,
            feat=args.features,
            sl=args.seq_len,
            ll=args.label_len,
            pl=args.pred_len,
            dm=args.d_model,
            el=args.e_layers,
            dl=args.d_layers,
            df=args.d_ff,
            fc=args.factor,
            eb=args.embed,
            dt=args.distil,
            des=args.des,
        )
        exp = Exp(args)
        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
