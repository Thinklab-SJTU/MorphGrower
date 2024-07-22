import torch
import os
import json
import torch
import numpy as np
import scipy.sparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy

from model.model import ConditionalSeq2SeqVAE, ConditionalSeqDecoder, ConditionalSeqEncoder, reconstruction_loss
from model.tgnn import TGNN
from model.vmf_batch import vMF
from utils.data_utils import fix_seed, node_calculation, edge_calculation
from utils.utils import load_neurons, Tree, Node
from utils.log_utils import parser_generate_args
from scripts.measure_branch import angle_metric, branches_metric


def create_input(data, branches, offsets, e, pre_node, args):
    branches = np.array(branches)
    data_dim = 3
    masking_element = 0
    wind_l = args.wind_len
    src_len = args.max_src_length
    dst_len = args.max_dst_length

    s_shape = (wind_l, src_len, data_dim)
    padded_source = torch.ones(s_shape) * masking_element
    t_shape = (dst_len, data_dim)
    target_l = torch.ones(t_shape) * masking_element
    target_r = torch.ones(t_shape) * masking_element

    real_wind_len, seq_len = 0, []
    prefix, targets, _ = data
    for idx, branch_id in enumerate(prefix[-wind_l:]):
        branch = torch.from_numpy(branches[branch_id])
        branch_l = len(branch)
        padded_source[idx][:branch_l] = branch
        real_wind_len += 1
        seq_len.append(branch_l)

    while len(seq_len) != wind_l:
        seq_len.append(0)
    seq_len = torch.LongTensor(seq_len)

    # teaching_force = 0, so these can be ignored
    target_l = torch.from_numpy(branches[targets[0]]).to(torch.float32)
    target_r = torch.from_numpy(branches[targets[1]]).to(torch.float32)
    target_len = torch.tensor([dst_len,dst_len])

    new_index = prefix[-1]

    node = torch.from_numpy(branches[pre_node[new_index]])
    node = node.to(torch.float32)

    # edge
    m, n = np.ix_(pre_node[new_index],pre_node[new_index])
    edge = e[m,n].tocoo()

    padded_source = torch.stack([padded_source])
    target_l = torch.stack([target_l])
    target_r = torch.stack([target_r])
    real_wind_len = torch.tensor([real_wind_len])
    seq_len = torch.stack([seq_len])
    target_len = torch.stack([target_len])

    _node = [node]
    offset = []
    offset += [0]*(len(node))

    if offset == []:
        offset = torch.tensor([])
    elif offset[-1] != 0:
        offset.append(0)
        _node.append(torch.zeros((1,32,3)))
    offset = torch.tensor(offset)
    _node = torch.concat(_node, dim=0)

    _edge = scipy.sparse.block_diag(mats=[edge])
    layer = _edge.data
    row = _edge.row
    col = _edge.col
    _data = np.ones(layer.shape)
    if _edge.shape[0]==0 or _edge.shape[1]==0:
        M=0
    else:
        M=_edge.max()
    _shape = (int(M+1), _edge.shape[0], _edge.shape[1])
    _edge = torch.sparse_coo_tensor(torch.tensor(np.vstack([layer,row,col])).to(torch.long), torch.tensor(_data), _shape)

    return (offsets[targets[0]],padded_source, target_l, target_r, real_wind_len, seq_len, target_len, _node, offset, _edge)

def generate_a_tree(neuron, model, args, radius=0.2, type_=1):
    branches, offsets, dataset, layer, nodes = neuron.fetch_branch_seq(align=args.align, move=True, need_angle=args.sort_angle, need_length=args.sort_length)
    e = edge_calculation(dataset, size = len(branches))
    pre_node = node_calculation(layer, nodes)

    new_branches = deepcopy(branches)
    new_branches = np.array(new_branches)
    queue, lf = [], 0
    root_pos = neuron.nodes[0].data['pos']
    print(root_pos)
    soma_n = Node(Idx=0, data={'pos': root_pos, 'radius': radius, 'type': type_})
    std_soma_n = Node(Idx=0, data={'pos': root_pos, 'radius': radius, 'type': type_})
    new_tree = Tree(root=soma_n)
    std_tree = Tree(root=std_soma_n)

    bid2data_dict = {}
    bid2tid_dict = {}
    std_bid2tid_dict = {}
    notSoma = set()
    counter = 0
    for idx, data in enumerate(dataset):
        prefix, targets, _ = data
        bid2data_dict[prefix[-1]] = data
        notSoma.add(prefix[-1])
        notSoma.add(targets[0])
        notSoma.add(targets[1])
        if (len(prefix)==1):
            branch = new_branches[prefix[-1]]
            curr_idx = 0
            for x in range(1,len(branch)):
                curr_idx = new_tree.add_node(
                    father_idx=curr_idx,
                    data={'pos': branch[x]+root_pos, 'type': type_, 'radius': radius}
                ).Idx
            bid2tid_dict[prefix[-1]] = curr_idx
            #print("---",data,"---")
            counter+=1
            queue.append((data,0,np.array(offsets[targets[0]])))

            std_curr_idx = 0
            for x in range(1,len(branch)):
                std_curr_idx = std_tree.add_node(
                    father_idx=std_curr_idx,
                    data={'pos': branch[x]+root_pos, 'type': type_, 'radius': radius}
                ).Idx
            std_bid2tid_dict[prefix[-1]] = std_curr_idx
    type_ = 2
    print(notSoma)
    for bid, br in enumerate(branches):
        if not bid in notSoma:
            if (offsets[bid]==[0.,0.,0.]).all():
                curr_idx = 0
                for x in range(1,len(br)):
                    curr_idx = new_tree.add_node(
                        father_idx=curr_idx,
                        data={'pos': br[x]+root_pos, 'type': type_, 'radius': radius}
                    ).Idx
                counter += 1

                std_curr_idx = 0
                for x in range(1,len(br)):
                    std_curr_idx = std_tree.add_node(
                        father_idx=std_curr_idx,
                        data={'pos': br[x]+root_pos, 'type': type_, 'radius': radius}
                    ).Idx
    N_stems = counter
    generate_loss = 0
    generate_count = 0
    depth_has_seen = 0
    neuron_layers = []
    generated_loss_layers = [0]
    std_neuron_layers = []
    counter1 = counter

    pdl, spl, dtwl, anglel = [], [], [], []
    while lf < len(queue):
        data, depth, new_offset = queue[lf]
        new_offset = new_offset
        if depth>depth_has_seen:
            depth_has_seen = depth
            if generate_count>0:
                generated_loss_layers.append(generate_loss/generate_count)
        _prefix, targets, _ = data

        offset, prefix, target_l, target_r, window_len, seq_len, target_seq_len, node, target_offset, edge = create_input(data,new_branches,offsets, e, pre_node , args)
        prefix, target_l, target_r, node, target_offset, edge = prefix.to(device), target_l.to(device), target_r.to(device), node.to(device), target_offset.to(device), edge.to(device)

        lf += 1
        with torch.no_grad():
            output_l, output_r, h, Z = model(prefix,seq_len,window_len,target_l,target_r,target_seq_len,node,target_offset,edge, teacher_force=args.teaching, need_gauss=args.need_gauss)
            loss1 = reconstruction_loss(output_l, target_l) + reconstruction_loss(output_r, target_r)
            output_l = output_l[0].cpu()
            output_r = output_r[0].cpu()

        generate_count += 1
        generate_loss += loss1.item()
        new_branches[targets[0]] = output_l.numpy()
        new_branches[targets[1]] = output_r.numpy()
        pd, sp, dtw = branches_metric(branches[targets[0]], new_branches[targets[0]])
        pdl.append(pd), spl.append(sp), dtwl.append(dtw)
        pd, sp, dtw = branches_metric(branches[targets[1]], new_branches[targets[1]])
        pdl.append(pd), spl.append(sp), dtwl.append(dtw)
        angle_ = angle_metric(branches[targets[0]], new_branches[targets[0]], branches[targets[1]], new_branches[targets[1]])
        anglel.append(angle_)
        def add_branch_to_tree(output, target, curr_idx, tree, bid2tid_dict, offset, root_pos):
            for x in range(1, len(output)):
                curr_idx = tree.add_node(
                    father_idx=curr_idx,
                    data={'pos': output[x].numpy()+offset+root_pos, 'type': type_, 'radius': radius}
                ).Idx
            bid2tid_dict[target] = curr_idx

        # left generated branch
        add_branch_to_tree(output=output_l,target=targets[0],curr_idx = bid2tid_dict[_prefix[-1]],tree=new_tree,bid2tid_dict=bid2tid_dict,offset=new_offset,root_pos=root_pos)
        target_data = bid2data_dict.get(targets[0])
        if target_data != None:
            queue.append((target_data,depth+1,new_offset+output_l[-1].numpy()))

        # right generated branch
        add_branch_to_tree(output=output_r,target=targets[1],curr_idx = bid2tid_dict[_prefix[-1]],tree=new_tree,bid2tid_dict=bid2tid_dict,offset=new_offset,root_pos=root_pos)
        target_data = bid2data_dict.get(targets[1])
        if target_data != None:
            queue.append((target_data,depth+1,new_offset+output_r[-1].numpy()))

        # std branches
        add_branch_to_tree(output=target_l[0].cpu(),target=targets[0],curr_idx = std_bid2tid_dict[_prefix[-1]],tree=std_tree,bid2tid_dict=std_bid2tid_dict,offset=offset,root_pos=root_pos)
        add_branch_to_tree(output=target_r[0].cpu(),target=targets[1],curr_idx = std_bid2tid_dict[_prefix[-1]],tree=std_tree,bid2tid_dict=std_bid2tid_dict,offset=offset,root_pos=root_pos)
        counter += 2
    print("***************")
    metric = np.mean(pdl), np.mean(spl), np.mean(dtwl), np.mean(anglel)
    print(metric)
    print(len(branches))
    print(len(dataset))
    print('N_stems', N_stems, neuron.N_stems())
    print(counter)
    print(counter-counter1)
    print("***************")
    if generate_count>0:
        generated_loss_layers.append(generate_loss/generate_count)
    if args.generate_layers==-1:
        return new_tree,std_tree, generate_loss, generate_count, metric
    else:
        return neuron_layers[:args.generate_layers], std_neuron_layers[:args.generate_layers], generate_loss, generate_count, generated_loss_layers[:args.generate_layers]

if __name__ == '__main__':
    args = parser_generate_args()
    print(args)
    fix_seed(args.seed)

    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')
    device = torch.device(device)
    weight = torch.load(args.model_path, map_location=torch.device(device))
    weight = weight['VAE']

    if args.model_type=='lstm':
        hidden = args.dim
        dropout = 0
        encoder = ConditionalSeqEncoder(3, hidden, hidden, dropout=dropout)
        decoder = ConditionalSeqDecoder(3, hidden, hidden, dropout=dropout)
        tgnn = TGNN(args.tgnn_size, args.tgnn_in_channel, args.tgnn_emb_channel, args.tgnn_out_channel)
        distribution = vMF(hidden, kappa=args.kappa, device=device)
        VAE = ConditionalSeq2SeqVAE(encoder, decoder, distribution, tgnn, device=device,forgettable=args.forgettable, remove_global=args.remove_global=='True', remove_path=args.remove_path=='True')
        VAE.to(device)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        result = {'Total': total_num, 'Trainable': trainable_num}
        print("Net:{} Total:{:.3f} Traing:{:.3f}".format(args.model_type, result["Total"], result['Trainable']))
        exit()

    VAE.load_state_dict(weight)
    VAE = VAE.eval()

    neurons, reidx = load_neurons(args.data_dir, scaling=args.scaling, return_reidx=True)

    log_path = args.model_path.replace('model','log').replace('.pth','.json')
    log = json.load(open(log_path))
    seed = log['args']['seed']
    last_dir = log_path.split('/')[-2]
    out_dir = os.path.join(args.output_dir, last_dir.split('-')[-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(last_dir)

    log_file = os.path.join(out_dir, str(args.seed)+'.json')
    log_reidx = dict(zip(log['reidx'].values(), log['reidx'].keys()))

    with open(log_file, 'w') as Fout:
        json.dump(log, Fout, indent=4)
    def get_losses_and_draw(log, ax, part):
        losses = log['losses'][part]
        res_loss = []
        for loss in losses:
            res_loss.append(loss['total'])

        ax.set_ylabel('loss')
        ax.set_xlabel(f'{part} loss')
        ax.plot(res_loss)
        return res_loss

    swc_dir = os.path.join(out_dir, 'swc')
    if not os.path.exists(swc_dir):
        os.makedirs(swc_dir)
    fig_path = os.path.join(out_dir, f'loss.png')
    figure, axes = plt.subplots(3, 1, sharex=False, sharey=False)
    axes = axes.flatten()
    train_losses = get_losses_and_draw(log, axes[0], 'train')
    valid_loss = get_losses_and_draw(log, axes[1],'valid')
    test_loss = get_losses_and_draw(log, axes[2],'test')
    plt.savefig(fig_path)

    log['generate_loss'] = {'train':[],'test-valid':[]}
    if args.in_one_graph:
        for split in ['train','test-valid']:
            dir_name = os.path.join(out_dir, split, 'all')
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    else:
        for split in ['train','test-valid']:
            dir_name = os.path.join(out_dir, split, args.projection_3d)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    _test_valid_split = [log['data_split']['test'], log['data_split']['valid']]
    test_valid_split = [x for item in _test_valid_split for x in item]
    train_split = log['data_split']['train']
    log_split = {'test-valid':test_valid_split, 'train':train_split, 'test':log['data_split']['test'], 'valid':log['data_split']['valid']}

    all_save = []
    for split in ['test']:
        total_loss = 0
        total_count = 0
        data_split = log_split[split]
        metric_list = {}
        if args.short!=None:
            data_split = data_split[:args.short]
        for _,idx in enumerate(tqdm(data_split)):
            neu = neurons[reidx[log_reidx[idx]]]
            print(log_reidx[idx])
            neu = deepcopy(neurons[idx])
            if args.in_one_graph:
                fig = plt.figure(figsize=(8, 6))
                grid = plt.GridSpec(2, 5, wspace=1, hspace=0.1)
                ax_3d = fig.add_subplot(grid[:,:2], projection='3d')
                # xy, xz ,yz
                ax_gt = [fig.add_subplot(grid[0,2]),fig.add_subplot(grid[0,3]),fig.add_subplot(grid[0,4])]
                ax_generated = [fig.add_subplot(grid[1,2]),fig.add_subplot(grid[1,3]),fig.add_subplot(grid[1,4])]
                for ax in ax_gt:
                    ax.set_aspect('equal')
                for ax in ax_generated:
                    ax.set_aspect('equal')

            if args.generate_layers == -1:
                new_neuron, std_neuron, generate_loss, generate_count, metric = generate_a_tree(
                    neu, VAE,
                    args
                )
                df = new_neuron.to_swc(scaling=10.)
                df.to_csv(os.path.join(swc_dir,log_reidx[idx]),header=None, index=None,sep=' ')
                metric_list[log_reidx[idx]] = metric
                if args.only_swc:
                    continue
                if not args.in_one_graph:
                    # draw 3d figure
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    new_neuron.draw_3d(ax=ax)
                    fig_name = "3d_"+log_reidx[idx].replace('.swc','.png')
                    fig_file = os.path.join(out_dir, split, args.projection_3d, fig_name)
                    plt.savefig(fig_file,bbox_inches='tight',format='png')
                    fig_file = fig_file.replace('png','pdf')
                    plt.savefig(fig_file,bbox_inches='tight',format='pdf')
                    print(fig_file)
                else:
                    new_neuron.draw_3d(ax=ax_3d)
                    neurons[idx].draw_3d(ax=ax_3d,axon_color='darkblue')
            else:
                new_neuron, std_neuron, generate_loss, generate_count, layers_loss = generate_a_tree(
                    neu, VAE,
                    args
                )
            total_loss += generate_loss
            total_count += generate_count

            def draw(neuron_file, neu,new_neuron, projection='xy'):
                if args.generate_layers==-1:
                    figure, axes = plt.subplots(1, 1, sharex=False, sharey=False)
                    plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                    axes.set_aspect('equal', adjustable='box')
                    neu.draw_2d(ax=axes, projection=projection)
                    fig_name = neuron_file.replace('.swc','_std.png')
                    fig_file = os.path.join(out_dir, split, projection, fig_name)
                    plt.axis('square')
                    plt.savefig(fig_file,bbox_inches='tight',format='png')
                    fig_file = fig_file.replace('.png','.pdf')
                    plt.savefig(fig_file,bbox_inches='tight',format='pdf')
                    print(fig_file)

                    figure, axes = plt.subplots(1, 1, sharex=False, sharey=False)
                    plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                    axes.set_aspect('equal', adjustable='box')
                    new_neuron.draw_2d(ax=axes, projection=projection)
                    fig_name = neuron_file.replace('.swc','_generated.png')
                    fig_file = os.path.join(out_dir, split, projection, fig_name)
                    plt.axis('square')
                    plt.savefig(fig_file,bbox_inches='tight',format='png')
                    fig_file = fig_file.replace('.png','.pdf')
                    plt.savefig(fig_file,bbox_inches='tight',format='pdf')
                else:
                    figure, axes = plt.subplots(2,args.generate_layers, sharex=True, sharey=True)
                    axes = axes.flatten()
                    for ax in axes:
                        ax.set_aspect('equal')
                    num = 0
                    for neu in std_neuron:
                        neu.draw_2d(ax=axes[num], projection = projection)
                        num += 1
                    for neu in new_neuron:
                        neu.draw_2d(ax=axes[num], projection = projection)
                        num += 1

            if args.in_one_graph:
                for projection,ax in zip(['xy','xz','yz'],ax_gt):
                    neurons[idx].draw_2d(ax=ax,projection=projection)
                for projection,ax in zip(['xy','xz','yz'],ax_generated):
                    new_neuron.draw_2d(ax=ax,projection=projection)
                fig_name = log_reidx[idx].replace('.SWC','.png')
                fig_name = fig_name.replace('.swc','.png')
                fig_file = os.path.join(out_dir, split, 'all', fig_name)
                plt.savefig(fig_file,bbox_inches='tight',format='png')
                print(fig_file)
                fig_file = fig_file.replace('png','pdf')
                plt.savefig(fig_file,bbox_inches='tight',format='pdf')
            else:
                for projection in ['xy','xz','yz']:
                    dir_name = os.path.join(out_dir,split, projection)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    draw(neuron_file=log_reidx[idx],neu=neu,new_neuron=new_neuron,projection=projection)
            if (_+1)%10==0:
                log['generate_loss'][split].append({'idx':_+1,'loss':total_loss/total_count})
                with open(log_file, 'w') as Fout:
                    json.dump(log, Fout, indent=4)

        if (len(data_split))%10!=0 and total_count>0:
            log['generate_loss'][split].append({'idx':len(data_split),'loss':total_loss/total_count})
            with open(log_file, 'w') as Fout:
                json.dump(log, Fout, indent=4)

        metric_file = os.path.join(out_dir, "metric.json")
        pd, sp, dtw, angles = [], [], [], []
        for value in metric_list.values():
            pd.append(value[0]), sp.append(value[1]), dtw.append(value[2]), angles.append(value[3])
        metric_list["mean"] = [np.mean(pd), np.mean(sp), np.mean(dtw), np.mean(angles)]
        with open(metric_file, 'w') as Fout:
            json.dump(metric_list, Fout, indent=4)


