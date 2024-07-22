import json
import argparse
import random
import torch
import os
import time

from model.model import reconstruction_loss, BranchEncRnn, BranchDecRnn, RNNAE


from utils.data_utils import Seq2SeqDataset, get_seq_to_seq_fn

from utils.utils import load_neurons
from utils.data_utils import fix_seed, fetch_walk_fix_dataset

from torch.utils.data import DataLoader
from torch.optim import Adam
from training import pretrain_one_epoch, preeval_one_epoch
from model.model import reconstruction_loss, BranchEncRnn, BranchDecRnn, RNNAE


def create_log_dir(args):
    if args.detail_log_dir == '':
        name = [
            f'lr_{args.lr}', f'bs_{args.bs}', f'dp_{args.dropout}',
            f'dim_{args.dim}', f'seed_{args.seed}',
            f'teaching_{args.teaching}', f'early_stop_{args.early_stop}',
            'ordered' if args.ordered else 'reversed'
        ]
        args.detail_log_dir = '-'.join(name)
    return os.path.join(
        args.base_log_dir, f'scaling_{args.scaling}', args.detail_log_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Experiment for soma branch and axon backbone'
        'generation, using Seq2SeqVAE'
    )
    parser.add_argument(
        '--seed', default=2023, type=int,
        help='random seed for experiment'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='learning rate for training'
    )
    parser.add_argument(
        '--bs', type=int, default=128,
        help='batch size for training'
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help='the dropout ratio for training'
    )
    parser.add_argument(
        '--max_length', default=32, type=int,
        help='the max length for resample'
    )
    parser.add_argument(
        '--teaching', default=0.5, type=float,
        help='the tearching rate for training vae'
    )
    parser.add_argument(
        '--train_ratio', default=0.7, type=float,
        help='the ratio of neurons for training, (default 0.7)'
    )
    parser.add_argument(
        '--valid_ratio', default=0.15, type=float,
        help='the ratio of neurons for valid, (default 0.15)'
    )
    parser.add_argument(
        '--data_dir', required=True, type=str,
        help='the path containing swc datas'
    )
    parser.add_argument(
        '--dim', default=64, type=int,
        help='the hidden dim of model, (default 64)'
    )
    parser.add_argument(
        '--device', default=-1, type=int,
        help='the gpu id for training, minus for cpu'
    )
    parser.add_argument(
        '--epochs', default=200, type=int,
        help='the number of epochs for training'
    )
    parser.add_argument(
        '--base_log_dir', type=str, default='log',
        help='the dir containg log, detailed as the dataset name'
    )
    parser.add_argument(
        '--detail_log_dir', type=str, default='',
        help='the detail log dir for logging, containg all parameters'
    )
    parser.add_argument(
        '--scaling', default=1, type=float,
        help='the scale the scale down the coordiante, (default: 1)'
    )

    parser.add_argument(
        '--early_stop', default=0, type=int,
        help='the compare epochs for early stop'
        'will be ignored when the number is less than 3'
    )
    parser.add_argument(
        '--model_path', help='the path of pretrained model',
        default='', type=str
    )
    parser.add_argument(
        '--ordered', action='store_true',
        help='if this option is chosen, dont reverse the source sequence'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='evaluate the well trained model'
    )

    args = parser.parse_args()
    fix_seed(args.seed)
    log_dir = create_log_dir(args)
    timestamp = time.time()
    print(args)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    neurons, reidx = load_neurons(
        args.data_dir, return_reidx=True,
        scaling=args.scaling
    )
    print('[INFO] neuron loaded')

    all_idx = list(range(len(neurons)))
    random.shuffle(all_idx)

    assert args.train_ratio + args.valid_ratio < 1, \
        'there should be a part for test set'

    assert args.train_ratio > 0 and args.valid_ratio > 0,\
        'there should be samples in train and valid set'

    train_num = int(args.train_ratio * len(neurons))
    valid_num = int(args.valid_ratio * len(neurons))
    train_idx = all_idx[:train_num]
    valid_idx = all_idx[train_num: train_num + valid_num]
    test_idx = all_idx[train_num + valid_num:]

    train_set = fetch_walk_fix_dataset(
        neurons=[neurons[t] for t in train_idx], verbose=True,
        seq_len=args.max_length, reverse=not args.ordered
    )
    valid_set = fetch_walk_fix_dataset(
        neurons=[neurons[t] for t in valid_idx], verbose=True,
        seq_len=args.max_length, reverse=not args.ordered
    )
    test_set = fetch_walk_fix_dataset(
        neurons=[neurons[t] for t in test_idx], verbose=True,
        seq_len=args.max_length, reverse=not args.ordered
    )

    train_loader = DataLoader(
        train_set, batch_size=args.bs, shuffle=True,
        collate_fn=get_seq_to_seq_fn(masking_element=0, output_dim=3)
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.bs, shuffle=False,
        collate_fn=get_seq_to_seq_fn(masking_element=0, output_dim=3)
    )
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=get_seq_to_seq_fn(masking_element=0, output_dim=3)
    )

    hidden = args.dim
    branch_enc = BranchEncRnn(3, hidden, hidden, dropout=args.dropout)
    branch_dec = BranchDecRnn(3, hidden, hidden, dropout=args.dropout)
    model = RNNAE(branch_enc, branch_dec).to(device)

    if args.model_path != '':
        weight = torch.load(args.model_path, map_location=device)
        VAE.load_state_dict(weight['whole'])
        print(f'[INFO] use model in {args.model_path} as base')

    if args.test:
        assert args.model_path != '', 'path of model weight should be provided'
        test_recon_loss = preeval_one_epoch(
            test_loader, VAE, reconstruction_loss, device
        )
        print('[INFO] test result:')
        print('reconstruction loss: ', test_recon_loss)

        exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data_split = {'train': train_idx, 'test': test_idx, 'valid': valid_idx}
    losses = {'train': [], 'test': [], 'valid': []}
    log_info = {
        'reidx': reidx, 'data_split': data_split,
        'args': args.__dict__, 'losses': losses
    }
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'log-{timestamp}.json')
    model_file = os.path.join(log_dir, f'model-{timestamp}.pth')

    best_ep, best_perf = None, None

    for epoch in range(args.epochs):
        print(f'[INFO] training on epoch {epoch}')
        train_loss = pretrain_one_epoch(
            train_loader, model, optimizer, reconstruction_loss,
            device, teaching=args.teaching
        )
        log_info['losses']['train'].append({'reconstruction': train_loss})
        print(f'[INFO] evaluting and testing on epoch {epoch}')
        val_recon_loss = preeval_one_epoch(
            valid_loader, model, reconstruction_loss, device
        )
        test_recon_loss = preeval_one_epoch(
            test_loader, model, reconstruction_loss, device
        )
        log_info['losses']['valid'].append({'reconstruction': val_recon_loss})
        log_info['losses']['test'].append({'reconstruction': test_recon_loss})
        if best_perf is None or val_recon_loss < best_perf:
            best_ep, best_perf = epoch, val_recon_loss
            torch.save({
                'whole': model.state_dict(),
                'branch_enc': model.encoder.state_dict(),
                'branch_dec': model.decoder.state_dict()
            }, model_file)
        print('[RESULT]')
        print('[TRAIN]', log_info['losses']['train'][-1])
        print('[VALID]', log_info['losses']['valid'][-1])
        print('[TEST]', log_info['losses']['test'][-1])
        with open(log_file, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if train_loss == float('nan'):
            print('[INFO] break because there is nan in loss')
            break

        if args.early_stop >= 3 and epoch >= args.early_stop:
            start = epoch - args.early_stop
            assert start >= 0, 'Invalid Start point'
            loss_seq = [
                x['reconstruction'] for x in
                log_info['losses']['valid'][start:]
            ]
            if all([x > loss_seq[0] for x in loss_seq[1:]]):
                break
