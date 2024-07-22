import torch
from torch_geometric.nn import MessagePassing, global_mean_pool

class TGNNconv(MessagePassing):
    def __init__(self, in_channels, emd_channels, out_channels, gru):
        super().__init__(aggr='mean')
        self.linear = torch.nn.Linear(in_channels, emd_channels)
        self.gru = gru
        self.norm = torch.nn.ReLU()
        self.in_channel = in_channels
        self.emd_channel = emd_channels

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [layer, 2, E[layer]]
        # encoder x
        x = self.linear(x)
        # propagating messages
        if edge_index._indices().shape[1]:
            for i in range(len(edge_index)):
                # get hidden messsage
                h_hat = self.propagate(edge_index=edge_index[i]._indices(), size=(x.size(0), x.size(0)), x=x)
                x = h_hat
        # output
        return self.norm(x)

    def message(self, x_j):
        return x_j

    def update(self, h, x):
        h = h.reshape(1, x.shape[0], x.shape[1])
        x = x.reshape(1, x.shape[0], x.shape[1])
        # use gru layer
        output, h_hat = self.gru(x, h)
        mask = (h == 0)
        h_hat[mask] = x[mask]
        return torch.squeeze(h_hat)


class TGNN(torch.nn.Module):
    def __init__(self, size, in_channel, emd_channel, out_channel):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(size, size, kernel_size=3, padding=0, stride=1)
        emd_channel = out_channel
        self.gru = torch.nn.GRU(emd_channel, out_channel)
        self.conv1 = TGNNconv(in_channel, emd_channel, out_channel, self.gru)
        self.conv2 = TGNNconv(out_channel, emd_channel, out_channel, self.gru)
        self.size = out_channel
        self.in_channel = in_channel
        self.emd_channel = emd_channel

    def forward(self, x, offset, edge):
        # x is [N*32*3], edge is [D*N*N]
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)
        e1 = self.conv1(x, edge)
        e2 = self.conv2(e1, edge)
        out = global_mean_pool(e2, offset)
        return out
