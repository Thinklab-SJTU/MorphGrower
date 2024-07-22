import json
import random
import torch
import os
import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from model.model import ConditionalSeq2SeqVAE, ConditionalSeqDecoder, ConditionalSeqEncoder,reconstruction_loss
from model.tgnn import TGNN
from model.vmf_batch import vMF
from utils.data_utils import fix_seed, load_weight, my_collate, tree_construction, ConditionalPrefixSeqDataset
from utils.utils import load_neurons
from utils.log_utils import parse_train_args, create_log_dir
from scripts.training import train_conditional_one_epoch, evaluateCVAE


def createDataset(neurons, neuron_files, args):
    branches, offsets, dataset, Tree = [],[],[],[]
    cnt = 0
    for neuron in neurons:
        print(cnt, neuron_files[cnt])
        cnt += 1
        single_branches, single_offsets, single_dataset, single_layer, single_node = neuron.fetch_branch_seq(align=args.align, move=True, need_length=args.sort_length, need_angle=args.sort_angle)
        for i in range(len(single_branches)):
            br = single_branches[i]
            if len(br) != args.align:
                print("alert")
                single_branches[i] = br[:args.align]
        single_Tree = tree_construction(single_branches, single_dataset, single_layer, single_node)
        # re-label dataset
        prefix_len = len(branches)
        new_single_dataset = []
        for data in single_dataset:
            prefix_array = list(x+prefix_len for x in data[0])
            target_tuple = tuple(x+prefix_len for x in data[1])
            new_single_dataset.append((prefix_array,target_tuple,data[2]))

        branches.extend(single_branches)
        dataset.extend(new_single_dataset)
        Tree.extend(single_Tree)
        offsets.extend(single_offsets)
    return ConditionalPrefixSeqDataset(branches,dataset,args.max_length,args.max_length,args.data_dim,args.wind_len,Tree)

if __name__ == '__main__':
    ############### log init ###############
    args = parse_train_args()
    fix_seed(args.seed)
    log_dir = create_log_dir(args)
    timestamp = time.time()
    print(args)

    ############### device set ###############
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    print("----------------")
    print(device)
    print("----------------")
    ############### dadtaset preparation ###############
    neurons, reidx = load_neurons(args.data_dir, scaling=args.scaling, return_reidx=True)
    neurons, neuron_files = load_neurons(args.data_dir, scaling=args.scaling, return_filelist=True)
    print('[INFO] neuron loaded')
    all_idx = list(range(len(neurons)))

    assert args.train_ratio + args.valid_ratio < 1, \
        'there should be a part for test set'

    assert args.train_ratio > 0 and args.valid_ratio > 0,\
        'there should be samples in train and valid set'
    if args.before_log_dir == '':
        random.shuffle(all_idx)
        train_num = int(args.train_ratio * len(neurons))
        valid_num = int(args.valid_ratio * len(neurons))
        train_idx = all_idx[:train_num]
        valid_idx = all_idx[train_num: train_num + valid_num]
        test_idx = all_idx[train_num + valid_num:]
    else:
        log = json.load(open(args.before_log_dir))
        train_idx = log['data_split']['train']
        valid_idx = log['data_split']['valid']
        test_idx = log['data_split']['test']

    print(len(train_idx))
    train_set = createDataset(neurons=[neurons[t] for t in train_idx],neuron_files=[neuron_files[t] for t in train_idx],args=args)
    print(len(valid_idx))
    valid_set = createDataset(neurons=[neurons[t] for t in valid_idx],neuron_files=[neuron_files[t] for t in valid_idx],args=args)
    print(len(test_idx))
    test_set = createDataset(neurons=[neurons[t] for t in test_idx],neuron_files=[neuron_files[t] for t in test_idx],args=args)

    train_loader = DataLoader(train_set, args.bs, shuffle=True,collate_fn=my_collate)
    valid_loader = DataLoader(valid_set, args.bs, shuffle=False,collate_fn=my_collate)
    test_loader = DataLoader(test_set, args.bs, shuffle=False,collate_fn=my_collate)

    ############### model create ###############
    if args.model_type=='lstm':
        hidden = args.dim
        dropout = args.dropout
        encoder = ConditionalSeqEncoder(3, hidden, hidden, dropout=dropout)
        decoder = ConditionalSeqDecoder(3, hidden, hidden, dropout=dropout)
        tgnn = TGNN(args.tgnn_size, args.tgnn_in_channel, args.tgnn_emb_channel, args.tgnn_out_channel)
        distribution = vMF(hidden, kappa=args.kappa, device=device)


        VAE = ConditionalSeq2SeqVAE(encoder, decoder, distribution, tgnn, device=device,forgettable=args.forgettable, remove_global=args.remove_global=='True', remove_path=args.remove_path=='True', new_model=args.new_model)
        VAE.to(device)

    if args.pretrained_path != None:
        weight = torch.load(args.pretrained_path, map_location=device)
        weight = weight['VAE']
        if args.dim == 64:
            weight.pop('state2latent.weight')
            weight.pop('state2latent.bias')
            weight.pop('latent2state.weight')
            weight.pop('latent2state.bias')
        VAE.load_state_dict(weight,strict=False)

    if args.model_path != '':
        weight = torch.load(args.model_path, map_location=device)
        load_weight(VAE, weight)
        print(f'[INFO] use model in {args.model_path} as base')

    if args.test:
        assert args.model_path != '', 'path of model weight should be provided'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        test_recon_loss = evaluateCVAE(test_loader, VAE, reconstruction_loss, device)
        print('[TEST]', test_recon_loss)
        exit()

    paras = VAE.parameters()
    optimizer = torch.optim.Adam(paras, lr=args.lr)
    regression_loss = torch.nn.MSELoss(reduction='sum')

    data_split = {'train': train_idx, 'test': test_idx, 'valid': valid_idx}
    losses = {'train': [], 'test': [], 'valid': []}
    log_info = {
        'data_split': data_split,
        'reidx': reidx,
        'args': args.__dict__,
        'losses': losses
    }

    train_losses, valid_losses, test_losses = [],[],[]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'log-{timestamp}.json')
    model_file = os.path.join(log_dir, f'model-{timestamp}.pth')

    fig_path = os.path.join(log_dir, f'loss-{timestamp}.png')
    best_ep, best_perf = None, None
    with open(log_file, 'w') as Fout:
        json.dump(log_info, Fout, indent=4)
    for epoch in range(args.epochs):
        print(f'[INFO] training on epoch {epoch}')
        train_recon_loss = train_conditional_one_epoch(
        train_loader, VAE, optimizer, reconstruction_loss, regression_loss,
        device, teaching=args.teaching)
        log_info['losses']['train'].append({
            'reconstruction': train_recon_loss,
            'total': train_recon_loss
        })
        print(f'[INFO] evaluting and testing on epoch {epoch}')
        valid_recon_loss = evaluateCVAE(
            valid_loader, VAE, reconstruction_loss, device
        )
        test_recon_loss = evaluateCVAE(
            test_loader, VAE, reconstruction_loss, device,
        )
        train_losses.append(train_recon_loss)
        valid_losses.append(valid_recon_loss)
        test_losses.append(test_recon_loss)
        log_info['losses']['valid'].append({
            'reconstruction': valid_recon_loss,
            'total': valid_recon_loss
        })
        log_info['losses']['test'].append({
            'reconstruction': test_recon_loss,
            'total': test_recon_loss
        })

        if best_perf is None or valid_recon_loss < best_perf:
            best_ep, best_perf = epoch, valid_recon_loss
            torch.save({
                'VAE': VAE.state_dict()
            }, model_file)

        print('[RESULT]')
        print('[TRAIN]', log_info['losses']['train'][-1])
        print('[VALID]', log_info['losses']['valid'][-1])
        print('[TEST]', log_info['losses']['test'][-1])
        with open(log_file, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if args.early_stop >= 3 and epoch >= args.early_stop:
            start = epoch - args.early_stop
            assert start >= 0, 'Invalid Start point'
            loss_seq = [
                x['reconstruction'] for x in
                log_info['losses']['valid'][start:]
            ]
            if all([x > loss_seq[0] for x in loss_seq[1:]]):
                break

    def draw_loss(ax, loss, part):
        ax.set_ylabel('loss')
        ax.set_xlabel(f'{part} loss')
        ax.plot(loss)

    figure, axes = plt.subplots(3, 1, sharex=False, sharey=False)
    axes = axes.flatten()
    draw_loss(axes[0],train_losses,'train')
    draw_loss(axes[1],valid_losses,'valid')
    draw_loss(axes[2],test_losses,'test')
    plt.savefig(fig_path)
