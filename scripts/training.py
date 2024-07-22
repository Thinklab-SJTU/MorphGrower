from tqdm import tqdm
import torch
import numpy as np

def train_conditional_one_epoch(
    dataloader, model, optimizer, recon_loss, reg_loss,
    device, teaching=0.5, alpha=1
):
    model = model.train()
    epoch_recon_loss, epoch_reg_loss, Nnum = 0, 0, 0
    for data in tqdm(dataloader):
        prefix, target_l, target_r, window_len, seq_len, target_seq_len, node, offset, edge = data
        prefix, target_l, target_r, node, offset, edge = prefix.to(device), target_l.to(device), target_r.to(device), node.to(device), offset.to(device), edge.to(device)
        output_l, output_r, h, Z = model(prefix,seq_len,window_len,target_l,target_r,target_seq_len,node,offset,edge,teacher_force=teaching)
        loss1 = recon_loss(output_l, target_l) + recon_loss(output_r, target_r)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        epoch_recon_loss += loss1.item()
        Nnum += len(prefix)
    return epoch_recon_loss / Nnum


def evaluateCVAE(
    data_loader, model, recon_loss, device
):
    model = model.eval()
    epoch_recon_loss = 0
    Nnum = 0
    for data in tqdm(data_loader):
        with torch.no_grad():
            prefix, target_l, target_r, window_len, seq_len, target_seq_len, node, offset, edge = data
            prefix, target_l, target_r, node, offset, edge = prefix.to(device), target_l.to(device), target_r.to(device), node.to(device), offset.to(device), edge.to(device)
            output_l, output_r, h, Z= model(prefix,seq_len,window_len,target_l,target_r,target_seq_len,node,offset,edge)
            loss1 = recon_loss(output_l, target_l) + recon_loss(output_r, target_r)
            epoch_recon_loss += loss1.item()
            Nnum += len(prefix)
    return epoch_recon_loss / Nnum



def pretrain_one_epoch(
    dataloader, model, optimizer, recon_loss,
    device, teaching=0.5,
):
    epoch_recon_loss,  model, Nnum = 0,  model.train(), 0
    for src, tgt, seq_len in tqdm(dataloader, ascii=True):
        walk_numbers = src.shape[0]
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)

        outputs = model(src, seq_len, target=tgt, teaching=teaching)

        recon = recon_loss(outputs, tgt)
        optimizer.zero_grad()
        recon.backward()
        optimizer.step()
        epoch_recon_loss += recon.item()
        Nnum += walk_numbers
    return epoch_recon_loss / Nnum


def preeval_one_epoch(dataloader, model, recon_loss, device):
    epoch_recon_loss, model, Nnum = 0, model.eval(), 0
    for src, tgt, seq_len in tqdm(dataloader, ascii=True):
        walk_numbers = src.shape[0]
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)
        # tgt shape [seq_len, batch_size, dim]
        with torch.no_grad():
            outputs = model(src, seq_len, teaching=0, target=tgt)
            recon = recon_loss(outputs, tgt)

            epoch_recon_loss += recon.item()
            Nnum += walk_numbers
    return epoch_recon_loss / Nnum
