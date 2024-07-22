import torch
import random
from torch.nn.utils.rnn import pack_padded_sequence


def reconstruction_loss(reconstructed_x, x, ignore_element=0):
    # reconstruction loss
    # x = [trg len, batch size * n walks, output dim] when tree major
    # x = [trg len, batch size, output dim] when batch major

    seq_len, batch_size, output_dim = x.shape
    mask = x[:, :, 0] != ignore_element
    rec_loss = 0
    # print(torch.all(mask != torch.isinf(x[:, :, 0])))
    for d in range(output_dim):
        # print(reconstructed_x[:, :, d][mask])
        # print(x[:, :, d][mask])
        rec_loss += torch.nn.functional.mse_loss(
            reconstructed_x[:, :, d][mask],
            x[:, :, d][mask], reduction='sum'
        )
        # print(rec_loss)
    return rec_loss / output_dim

class ConditionalSeqEncoder(torch.nn.Module):
    # branch encoder
    # encode one branch into states of
    # hidden & cell (both [n_layers,hidden_dim])
    # Same as the SeqEncoder
    def __init__(
            self, input_dim, embedding_dim,
            hidden_dim, n_layers=2, dropout=0.5
    ):
        super(ConditionalSeqEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.coordinate2emb = torch.nn.Linear(input_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout
        )

    def forward(self, src, seq_len):
        input_seq = self.dropout_fun(self.coordinate2emb(src))
        packed_seq = pack_padded_sequence(
            input_seq, seq_len, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_seq)

        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell


class ConditionalSeqDecoder(torch.nn.Module):
    # Same as the SeqDecoder
    def __init__(
            self, output_dim, embedding_dim, hidden_dim,
            n_layers=2, dropout=0.5
    ):
        super(ConditionalSeqDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hid2coordinate = torch.nn.Linear(hidden_dim, output_dim)
        self.coordinate2emb = torch.nn.Linear(output_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout
        )

    def forward(self, init, hidden, cell):
        init = init.unsqueeze(0)
        # init = [1, batch_size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        embedding = self.dropout_fun(self.coordinate2emb(init))
        # print("embedding",embedding.shape)
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.hid2coordinate(output).squeeze(0)
        return prediction, hidden, cell


# target_len 重采样到 max_dst_len
def conditional_decode_seq(
        decoder, output_shape, init, hidden, cell,
        device, teacher_force=0.5, target=None
):
    if teacher_force > 0 and target is None:
        raise NotImplementedError(
            'require stadard sequence as input'
            'when using teacher force'
        )
    target_len, batch_size, output_dim = output_shape
    outputs = torch.zeros(output_shape).to(device)
    current, outputs[0] = init, init
    # print('init',init.shape, init)
    for t in range(1, target_len):
        output, hidden, cell = decoder(current, hidden, cell)
        # print('output',output.shape)
        outputs[t] = output
        current = target[t] if random.random() < teacher_force else output
        # print('current',current.shape)

    return outputs

class ConditionEncoder(torch.nn.Module):
    def __init__(self, branch_encoder, hidden_dim, n_layers=2, dropout=0.5):
        super(ConditionEncoder, self).__init__()
        self.branch_encoder = branch_encoder
        self.path_rnn = torch.nn.LSTM(
            branch_encoder.n_layers * branch_encoder.hidden_dim * 2,
            hidden_dim, n_layers, dropout=dropout
        )
        self.hidden_dim, self.n_layers = hidden_dim, n_layers

    def forward(self, prefix, seq_len, window_len):
        # prefix = [bs, window len, seq_len, data_dim]
        # seq_len = [bs, window len]
        # window_len = [bs]
        bs, wind_l, seq_l, input_dim = prefix.shape
        all_seq_len, all_seq = [], []
        for idx, t in enumerate(window_len):
            all_seq_len.extend(seq_len[idx][:t])
            all_seq.append(prefix[idx][:t])
        all_seq = torch.cat(all_seq, dim=0).permute(1, 0, 2)
        # print('[info] seq_shape', all_seq.shape, sum(window_len))

        h_branch, c_branch = self.branch_encoder(all_seq, all_seq_len)
        # print('[info] hshape', h_branch.shape, c_branch.shape)

        hidden_seq = torch.cat([h_branch, c_branch], dim=0)
        # print('[info] hidden_seq_shape', hidden_seq.shape)
        seq_number = sum(window_len)
        inter_dim = self.branch_encoder.n_layers * \
            self.branch_encoder.hidden_dim * 2
        hidden_seq = hidden_seq.transpose(0, 1).reshape(seq_number, -1)
        all_hidden = torch.zeros((bs, wind_l, inter_dim)).to(hidden_seq)
        curr_pos = 0
        for idx, t in enumerate(window_len):
            all_hidden[idx][:t] = hidden_seq[curr_pos: curr_pos + t]
            curr_pos += t
        assert curr_pos == seq_number, 'hidden vars dispatched error'

        all_hidden = all_hidden.permute(1, 0, 2)
        # print('[info] all hidden shape', all_hidden.shape, len(window_len))
        packed_wind = pack_padded_sequence(
            all_hidden, window_len, enforce_sorted=False
        )
        _, (h_path, c_path) = self.path_rnn(packed_wind)
        return h_path, c_path


class ConditionalSeq2SeqVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, distribution, tgnn, device, forgettable=None, remove_path=False,
                remove_global=False, new_model=False, dropout=0.1):
        super(ConditionalSeq2SeqVAE, self).__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.distribution = distribution
        self.new_model = new_model
        if new_model:
            self.condition_encoder = ConditionEncoder(self.encoder, self.encoder.hidden_dim, self.encoder.n_layers, dropout=dropout)
        # forgettable == None mean pooling
        # otherwise h_path[w] = forgettable * h_path[w-1]
        #                       + (1 - forgettable) * raw_embedding[w-1]
        self.forgettable = forgettable if forgettable != 0 else None

        print("**************************************")
        print(remove_global, remove_path)
        print("**************************************")

        self.tgnn = tgnn.to(device)
        self.global_dim = self.tgnn.size
        self.remove_global = remove_global
        self.remove_path = remove_path

        mean = torch.full([1,encoder.hidden_dim],0.0)
        std = torch.full([1,encoder.hidden_dim],1.0)
        self.gauss = torch.distributions.Normal(mean, std)

        self.state2latent = torch.nn.Linear(
            encoder.hidden_dim * encoder.n_layers * 6 + self.global_dim,
            distribution.lat_dim
        )
        self.latent2state_l = torch.nn.Linear(
            distribution.lat_dim + encoder.hidden_dim * encoder.n_layers * 2 + self.global_dim,
            decoder.hidden_dim * decoder.n_layers * 2
        )
        self.latent2state_r = torch.nn.Linear(
            distribution.lat_dim + encoder.hidden_dim * encoder.n_layers * 2 + self.global_dim,
            decoder.hidden_dim * decoder.n_layers * 2
        )
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "encoder and decoder must have equal number of layers!"

    def encode(self, src_l, seq_len_l, src_r, seq_len_r, h_path, h_global):
        hidden_l, cell_l = self.encoder(src_l, seq_len_l)
        n_layers, batch_size, hid_dim = hidden_l.shape
        states_l = torch.cat((hidden_l, cell_l), dim=0)
        # result states = [bs, 2*n_layers*hidden_dim]
        states_l = states_l.permute(1, 0, 2).reshape(batch_size, -1)

        hidden_r, cell_r = self.encoder(src_r, seq_len_r)
        states_r = torch.cat((hidden_r, cell_r), dim=0)
        # result states = [bs, 2*n_layers*hidden_dim]
        states_r = states_r.permute(1, 0, 2).reshape(batch_size, -1)

        # 拼接上 h_path
        # 拼接上TGN output h_global，shape = [bs,self.global_dim]
        states = torch.cat((states_l, states_r, h_path, h_global), dim=1)
        h = self.state2latent(states)
        tup, kld, vecs = self.distribution.build_bow_rep(h, n_sample=5)
        Z = torch.mean(vecs, dim=0)
        condition = h_global
        return h, Z, condition

    def _get_decoder_states(self, latent, batch_size, h_path, h_global, decode_left):
        # cat latent with h_path and h_global
        h = torch.cat((latent, h_path, h_global), dim=1)
        if decode_left:
            decoder_states = self.latent2state_l(h).reshape(batch_size, -1, 2)
        else:
            decoder_states = self.latent2state_r(h).reshape(batch_size, -1, 2)
        hidden_shape = (
            batch_size, self.decoder.hidden_dim, self.decoder.n_layers
        )
        hidden = decoder_states[:, :, 0].reshape(*hidden_shape)
        hidden = hidden.permute(2, 0, 1).contiguous()

        cell = decoder_states[:, :, 1].reshape(*hidden_shape)
        cell = cell.permute(2, 0, 1).contiguous()
        return hidden, cell

    def forward(self, prefix, seq_len, window_len, target_l, target_r, target_seq_len, node, offset, edge,
                teacher_force=0.5, need_gauss=False):
        # prefix = [bs, max window len, max seq len, data dim]
        # seq_len is the real seq len, seq_len = [bs, max window len]
        # window_len is the real window len, window_len = [bs]
        # encoder is just a branch encoder
        # target = [bs, max seq len, data dim]
        # target_seq_len = [bs,2]
        batch_size, max_wind_l, max_seq_l, input_dim = prefix.shape
        target_l_seq_len = target_seq_len[:, 0]
        target_r_seq_len = target_seq_len[:, 1]
        output_dim = self.decoder.output_dim

        # get h_path using pooling
        all_seq_len, all_seq = [], []
        for idx, t in enumerate(window_len):
            all_seq_len.extend(seq_len[idx][:t])
            all_seq.append(prefix[idx][:t])
            
        # all_seq = [max seq len, sum(window_len), data dim] after permute
        all_seq = torch.cat(all_seq, dim=0).permute(1, 0, 2)
        hiddens, cells = self.encoder(all_seq, all_seq_len)

        hidden_seq = torch.cat([hiddens, cells], dim=-1)

        # pooling to get h_path = [bs, n_layers*2*hidden_dim]
        if self.remove_path:
            h_path = torch.zeros([batch_size, hidden_seq.shape[0] * hidden_seq.shape[2]]).to(self.device)
        else:
            if self.new_model:
                cond_hidden, cond_cell = self.condition_encoder(prefix, seq_len, window_len)
                h_path = torch.cat([cond_hidden, cond_cell], dim=0)
                h_path = h_path.permute(1, 0, 2).reshape(batch_size, -1)
            else:
                h_path = []
                pre_cnt = 0

                def compress_path_embedding(y, forgettable, device):
                    if forgettable == None:
                        return torch.mean(y, 1)
                    else:
                        h = torch.zeros(y.shape[0], y.shape[2]).to(device)
                        for i in range(y.shape[1]):
                            h = forgettable * h + (1 - forgettable) * y[:, i, :]
                        return h

                for _, wl in enumerate(window_len):
                    h_path.append(
                        compress_path_embedding(hidden_seq[:, pre_cnt:pre_cnt + wl, :], self.forgettable, self.device))
                    pre_cnt += wl
                h_path = torch.stack(h_path).reshape(batch_size, -1)

        if self.remove_global:
            h_global = torch.zeros([batch_size, self.global_dim]).to(self.device)
        else:
            if offset.shape[0] == 0:
                h_global = torch.zeros([batch_size, self.global_dim]).to(self.device)
            else:
                node = node.permute(1, 0, 2)
                node_len = target_l_seq_len[0] * torch.ones(node.shape[1])
                hidden, cell = self.encoder(node, node_len)
                node = torch.cat((hidden, cell), dim=0)
                node = node.permute(1, 0, 2).reshape(hidden.shape[1], -1)
                h_global = self.tgnn(node, offset, edge)


        target_l = target_l.permute(1, 0, 2)
        target_r = target_r.permute(1, 0, 2)
        if need_gauss:
            h = 0
            Z = self.gauss.sample().to(self.device)
            Z = Z / torch.norm(Z)
        else:
            h, Z, condition = self.encode(target_l, target_l_seq_len, target_r, target_r_seq_len, h_path, h_global)
        hidden, cell = self._get_decoder_states(Z, batch_size, h_path, h_global, True)

        target_len = target_l.shape[0]

        output_l = conditional_decode_seq(
            self.decoder, (target_len, batch_size, output_dim),
            target_l[0], hidden, cell, self.device,
            teacher_force=teacher_force, target=target_l
        )
        output_l = output_l.permute(1, 0, 2)

        hidden, cell = self._get_decoder_states(Z, batch_size, h_path, h_global, False)

        target_len = target_r.shape[0]

        output_r = conditional_decode_seq(
            self.decoder, (target_len, batch_size, output_dim),
            target_r[0], hidden, cell, self.device,
            teacher_force=teacher_force, target=target_r
        )
        output_r = output_r.permute(1, 0, 2)
        return output_l, output_r, h, Z




class BranchEncRnn(torch.nn.Module):
    def __init__(
        self, input_dim, embedding_dim, hidden_dim,
        n_layers=2, dropout=0.5
    ):
        super(BranchEncRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.coordinate2emb = torch.nn.Linear(input_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def forward(self, src, seq_len, return_in_one=False):
        input_seq = self.dropout_fun(self.coordinate2emb(src))
        packed_seq = pack_padded_sequence(
            input_seq, seq_len, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_seq)

        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        #
        if not return_in_one:
            return hidden, cell
        else:
            batch_size = hidden.shape[1]
            answer = torch.cat([hidden, cell], dim=0)
            answer = answer.permute(1, 0, 2).reshape(batch_size, -1)
            return answer


class BranchDecRnn(torch.nn.Module):
    def __init__(
        self, output_dim, embedding_dim, hidden_dim,
        n_layers=2, dropout=0.5
    ):
        super(BranchDecRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hid2coordinate = torch.nn.Linear(hidden_dim, output_dim)
        self.coordinate2emb = torch.nn.Linear(output_dim, embedding_dim)
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def decode_a_step(self, init, hidden, cell):
        init = init.unsqueeze(0)
        # init = [1, batch_size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        embedding = self.dropout_fun(self.coordinate2emb(init))
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.hid2coordinate(output).squeeze(0)
        return prediction, hidden, cell

    def forward(
        self, hidden, cell, target_len=None, target=None,
        teaching=0.5, init=None
    ):
        if target_len is None and target is None:
            raise ValueError('the target_length should be specified')
        if init is None and target is None:
            raise ValueError('the start point should be specified')
        if teaching > 0 and target is None:
            raise NotImplementedError(
                'require stadard sequence as input'
                'when using teacher force'
            )

        if target_len is None:
            target_len = target.shape[0]
        if init is None:
            init = target[0]

        batch_size = hidden.shape[1]
        output_shape = (target_len, batch_size, self.output_dim)
        outputs = torch.zeros(output_shape).to(hidden.device)
        outputs[0] = init
        current = outputs[0].clone()
        for t in range(1, target_len):
            output, hidden, cell = self.decode_a_step(current, hidden, cell)
            outputs[t] = output
            current = target[t] if random.random() < teaching else output
        return outputs




class RNNAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src, seq_len, target=None, teaching=0.5,
        init=None, target_len=None
    ):
        hidden, cell = self.encoder(src, seq_len, return_in_one=False)
        return self.decoder(
            hidden, cell, target_len=target_len,
            target=target, init=init, teaching=teaching
        )