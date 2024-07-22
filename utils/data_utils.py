import numpy as np
import torch
import random
import scipy.sparse
from utils.utils import resample_branch_by_step

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_weight(model, weight, reg_model=None):
    model.load_state_dict(weight['VAE'])
    if 'regression' in weight and reg_model is not None:
        reg_model.load_state_dict(weight['regression'])

def edge_calculation(dataset, size=256):
    rows = []
    cols = []
    data = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i][1])):
            rows.append(dataset[i][1][j])
            cols.append(dataset[i][0][-1])
            data.append(len(dataset[i][0]))
    edge = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    return edge

def node_calculation(layer, node):
    pre_node = [[] for i in range(len(layer))]
    for i in range(len(layer)):
        for depth in range(layer[i][1]):
            pre_node[i] += node[layer[i][0]][depth]
    return pre_node

def tree_construction(branches, dataset, layer, nodes):
    branches = np.array(branches)
    e = edge_calculation(dataset, size=len(branches))
    pre_node = node_calculation(layer, nodes)
    tree = []
    for i in range(len(branches)):
        node = branches[pre_node[i]]
        m, n = np.ix_(pre_node[i], pre_node[i])
        edge = e[m, n].tocoo()
        tree.append({'edge': edge, 'node': node})
    return tree

def my_collate(data):
    padded_source = torch.stack([data[i][0] for i in range(len(data))], dim=0)
    target_l = torch.stack([data[i][1] for i in range(len(data))], dim=0)
    target_r = torch.stack([data[i][2] for i in range(len(data))], dim=0)
    real_wind_len = torch.stack([torch.tensor(data[i][3]) for i in range(len(data))], dim=0)
    seq_len = torch.stack([data[i][4] for i in range(len(data))], dim=0)
    target_len = torch.stack([data[i][5] for i in range(len(data))], dim=0)

    node = [data[i][6] for i in range(len(data))]
    offset = []
    for i in range(len(data)):
        offset += [i for j in range(len(data[i][6]))]

    if offset == []:
        offset = torch.tensor([])
    elif offset[-1] != len(data) - 1:
        offset.append(len(data) - 1)
        node.append(torch.zeros((1, 16, 3)))
    offset = torch.tensor(offset)
    node = torch.concat(node, dim=0)

    edge = scipy.sparse.block_diag(mats=[data[j][7] for j in range(len(data))])
    layer = edge.data
    row = edge.row
    col = edge.col
    data = np.ones(layer.shape)
    if edge.shape[0] == 0 or edge.shape[1] == 0:
        _max = 0
    else:
        _max = edge.max()
    shape = (int(_max + 1), edge.shape[0], edge.shape[1])
    edge = torch.sparse_coo_tensor(torch.tensor(np.vstack([layer, row, col])).to(torch.long), torch.tensor(data), shape)
    return (padded_source, target_l, target_r, real_wind_len, seq_len, target_len, node, offset, edge)

class ConditionalPrefixSeqDataset(torch.utils.data.Dataset):
    def __init__(
            self, branches, dataset,
            max_src_length, max_dst_length, data_dim, max_window_length, trees, max_size=128,
            masking_element=0, resample=True
    ):
        self.branches = branches
        self.dataset = dataset
        self.max_src_length = max_src_length
        self.max_dst_length = max_dst_length
        self.max_window_length = max_window_length
        self.data_dim = data_dim
        self.masking_element = masking_element
        self.resample = resample
        self.trees = trees
        self.max_size = max_size
        self.resampled_branches = [resample_branch_by_step(branch, self.max_dst_length, len(branch)) for branch in
                                branches] if self.resample else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        wind_l = self.max_window_length
        s_shape = (wind_l, self.max_src_length, self.data_dim)
        padded_source = torch.ones(s_shape) * self.masking_element
        t_shape = (self.max_dst_length, self.data_dim)
        target_l = torch.ones(t_shape) * self.masking_element
        target_r = torch.ones(t_shape) * self.masking_element

        real_wind_len, seq_len = 0, []
        prefix, targets, _ = self.dataset[index]
        for idx, branch_id in enumerate(prefix[-wind_l:]):
            branch = torch.from_numpy(self.branches[branch_id])
            branch_l = len(branch)
            padded_source[idx][:branch_l] = branch
            real_wind_len += 1
            seq_len.append(branch_l)

        while len(seq_len) != wind_l:
            seq_len.append(0)
        seq_len = torch.LongTensor(seq_len)
        if self.resample:
            target_l = torch.from_numpy(self.resampled_branches[targets[0]]).to(torch.float32)
            target_r = torch.from_numpy(self.resampled_branches[targets[1]]).to(torch.float32)
            target_len = torch.tensor([self.max_dst_length, self.max_dst_length])
        else:
            branch_l, branch_r = self.branches[targets[0]], self.branches[targets[1]]
            target_len = [len(branch_l), len(branch_r)]
            target_l[:target_len[0]] = torch.from_numpy(branch_l)
            target_r[:target_len[1]] = torch.from_numpy(branch_r)

        new_index = prefix[-1]
        node = torch.from_numpy(self.trees[new_index]['node'])
        node = node.to(torch.float32)
        edge = self.trees[new_index]['edge']
        return padded_source, target_l, target_r, real_wind_len, seq_len, target_len, node, edge



class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, reverse=True):
        super(Seq2SeqDataset, self).__init__()
        self.data, self.seq_len = data, seq_len
        self.reverse = reverse

    def __len__(self):
        return len(self.seq_len)

    def __getitem__(self, index):
        t_len = self.seq_len[index]
        target = self.data[index][:t_len]
        source = target[::-1] if self.reverse else target
        assert len(source) == len(target), \
            'source and target are not of the same length'
        return source, target, t_len


def get_seq_to_seq_fn(masking_element, output_dim):
    def col_fn(batch):
        max_len, batch_size, tls = max(x[2] for x in batch), len(batch), []
        pad_src = np.ones((batch_size, max_len, output_dim)) * masking_element
        pad_tgt = np.ones((batch_size, max_len, output_dim)) * masking_element

        for idx, (src, tgt, tl) in enumerate(batch):
            tls.append(tl)
            pad_src[idx, :tl] = src
            pad_tgt[idx, :tl] = tgt

        pad_src = torch.from_numpy(pad_src).float()
        pad_tgt = torch.from_numpy(pad_tgt).float()

        return pad_src, pad_tgt, tls
    return col_fn



def fetch_walk_fix_dataset(neurons, seq_len, reverse, verbose=False):
    all_walks = []
    for neu in (tqdm(neurons, ascii=True) if verbose else neurons):
        curr_walks = neu.fetch_all_walks()
        all_walks.extend([
            resample_branch_by_step(x, seq_len, len(x))
            for x in curr_walks
        ])
    seq_lens = [seq_len] * len(all_walks)
    return Seq2SeqDataset(data=all_walks, seq_len=seq_lens, reverse=reverse)