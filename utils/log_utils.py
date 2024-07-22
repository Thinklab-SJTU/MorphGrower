import os
import argparse

def str2bool(x: str):
    assert x == "False" or x == "True"
    if x == "True":
        return True
    else:
        return False

def create_log_dir(args):
    ablation = 'origin'
    if args.remove_path=='True':
        ablation = "remove_path"
        if args.remove_global=='True':
            ablation = "remove_both"
    elif args.remove_global=='True':
        ablation = "remove_global"

    if args.detail_log_dir == '':
        name = [
            f'lr_{args.lr}', f'bs_{args.bs}', f'dp_{args.dropout}',
            f'dim_{args.dim}', f'kappa_{args.kappa}',
            f'alpha_{args.alpha}', f'teaching_{args.teaching}',
            f'ds_type_{args.data_type}', f'early_stop_{args.early_stop}', f'align_{args.align}',
            'ordered' if args.ordered else 'reversed','mean_pooling' if args.forgettable==None else f'forgettable_{args.forgettable}',
            'initialized' if args.pretrained_path==None else 'load_pretrained', ablation
        ]
        args.detail_log_dir = '-'.join(name)
    return os.path.join(
        args.base_log_dir, ablation,'initialized' if args.pretrained_path==None else 'load_pretrained','mean_pooling' if args.forgettable==None else f'forgettable_{args.forgettable}', f'scaling_{args.scaling}', args.detail_log_dir, f'seed_{args.seed}'
    )

def add_basic_args(parser):
    parser.add_argument('--seed', default=2023, type=int,
                        help='random seed for experiment')
    parser.add_argument('--device', default=-1, type=int,
                        help='the gpu id for training, minus for cpu')
    return parser

def add_model_args(parser):
    parser.add_argument('--model_type', type=str, default='lstm')
    parser.add_argument('--new_model', type=str2bool, default=False)
    parser.add_argument('--remove_path', type=str , default='False')
    parser.add_argument('--remove_global', type=str, default=False)

    parser.add_argument('--data_dim',type=int,default=3)
    parser.add_argument('--dim', default=64, type=int,
                        help='the hidden dim of model, (default 64)')
    parser.add_argument('--tgnn_in_channel', default=256, type=int,
                        help = 'the tgnn in_channel dimmension')
    parser.add_argument('--tgnn_emb_channel', default=64, type=int,
                        help = 'the tgnn emb_channel dimmension')
    parser.add_argument('--tgnn_out_channel', default=64, type=int,
                        help = 'the tgnn out_channel dimmension')
    parser.add_argument('--tgnn_size', default=1, type=int,
                        help = 'the size of tgnn')
    parser.add_argument('--kappa', default=100, type=int,
                        help='the kappa for vMF distribution')
    parser.add_argument('--forgettable', default=None, type=float,
                        help='the rate that the CVAE model is forgettable for path encoding')
    return parser

def add_data_args(parser):
    parser.add_argument('--data_dir', required=True, type=str,
                        help='the path containing swc datas')
    parser.add_argument('--data_type', choices=['axon', 'dendrite', 'toy'], default='dendrite', type=str,
        help='the type of dataset, if it\'s axon, the dataset contains'
        'the trunk from axon cells, if it\'s dendrite, dataset contrains'
        'all branches from the cell, if it\'s toy, dataset contains all'
        'walks (from root to leaf) to train')

    parser.add_argument('--sort_length', type=str2bool , default=False)
    parser.add_argument('--sort_angle',type=str2bool,default=False)
    parser.add_argument('--window',type=int,default=15)
    parser.add_argument('--smooth',action="store_true", default=False)
    parser.add_argument('--cut',type=int,default=5)

    parser.add_argument('--max_length', default=32, type=int,
                        help='the max length for resample')
    parser.add_argument('--wind_len', default=4, type=int,
                        help='the max length of the prefix window')
    parser.add_argument('--align', default=32, type=int,
                        help = 'the step number to resample, need to be greater than 1')
    parser.add_argument('--scaling', default=10, type=float,
                        help='the scale the scale down the coordiante, (default: 1)')
    return parser

def add_train_args(parser):
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='pretrain parameter path of the pretrained encoder & decoder')
    parser.add_argument('--before_log_dir', type=str, default='',
                        help='thebefore log dir for logging, containg all parameters')
    parser.add_argument('--base_log_dir', type=str, default='log',
                        help='the dir containg log, detailed as the dataset name')
    parser.add_argument('--detail_log_dir', type=str, default='',
                        help='the detail log dir for logging, containg all parameters')
    parser.add_argument('--model_path',  default='', type=str,
                        help='the path of pretrained model')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size for training')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help='the dropout ratio for training')
    parser.add_argument('--early_stop', default=20, type=int,
                        help='the compare epochs for early stop'
                        'will be ignored when the number is less than 3')
    parser.add_argument('--epochs', default=200, type=int,
                        help='the number of epochs for training')

    parser.add_argument('--train_ratio', default=0.7, type=float,
                        help='the ratio of neurons for training, (default 0.7)')
    parser.add_argument('--valid_ratio', default=0.15, type=float,
                        help='the ratio of neurons for valid, (default 0.15)')
    parser.add_argument('--alpha', default=1, type=float,
                        help='the weight of regression loss')
    parser.add_argument('--ordered', action='store_true',
                        help='if this option is chosen, dont reverse the source sequence')
    parser.add_argument('--test', action='store_true',
                        help='evaluate the well trained model')
    parser.add_argument('--teaching', default=0.5, type=float,
                        help='the tearching rate for training vae')
    return parser

def add_generate_args(parser):
    parser.add_argument('--need_gauss',type=str2bool,default=False)
    parser.add_argument('--in_one_graph', default=False, type=bool)
    parser.add_argument('--only_swc',action='store_true')
    parser.add_argument('--projection_3d', default='xyz',type=str)
    parser.add_argument('--short', default=None, type=int)
    parser.add_argument('--teaching', default=0, type=float,
                        help='teaching force')
    parser.add_argument('--generate_layers', default=-1, type=int,
                        help='the layers to draw, recommended to be no more than 8. -1 for draw'
                        'whole neuron.')

    parser.add_argument('--max_window_length', default=8, type=int,
                        help='the max number of branches on prefix')
    parser.add_argument('--max_src_length', default=32, type=int,
                        help='the max length for generated branches')
    parser.add_argument('--max_dst_length', default=32, type=int,
                        help='--max length for generating branches')

    parser.add_argument('--model_path',type=str, required=True)
    parser.add_argument('--output_dir', required=True, type=str,
                        help='the dir containing results')
    return parser


def parse_train_args():
    parser = argparse.ArgumentParser(
        'Experiment for soma branch and axon backbone generation'
    )
    parser = add_basic_args(parser)
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_train_args(parser)
    args = parser.parse_args()
    return args

def parser_generate_args():
    parser = argparse.ArgumentParser('Generation of neuron truck and soma')
    parser = add_basic_args(parser)
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_generate_args(parser)
    args = parser.parse_args()
    return args

def parser_measure_args():
    parser = argparse.ArgumentParser('Measure')
    parser.add_argument('--data_path', required=True, type=str,
        help='random seed for experiment')
    parser.add_argument('--measure_type', choices=['measure', 'forest'], default='measure', type=str)
    parser.add_argument('--is_morph', default=False, type=str2bool)
    args = parser.parse_args()
    return args