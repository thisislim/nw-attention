import argparse
from multiprocessing.sharedctypes import Value
import os
import sys
import pickle
import time
import random
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import Levenshtein

from model.transformer import AttentionTransformerEncoder
from model.mha import MultiheadAttentionMapper
from model.cnn import CNNed
from model.pair_encoding import PairEncoding
from data_utils.data_handler import load_dataset, get_dataloader, load_metadata, load_dataset_list
from model.transformer import generate_pad_mask
from utils.math_and_loss import rmse, LossMeter
from utils.checkpoint import save_checkpoints, to_enc_dict, to_dec_dict

from torch.utils.tensorboard import SummaryWriter



def train(args):
    
    if torch.cuda.is_available() and args.cuda < torch.cuda.device_count():
        device = f'cuda:{args.cuda}'
    else:
        device = 'cpu'
    device = torch.device(device)
    print("# device: '{}'".format(device))

    torch.manual_seed(args.seed)
    
    dataset_list = load_dataset_list(args.data_path)

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/{args.task}_epch{args.epochs}_bs{args.batch_size}_{date}'
    writer = SummaryWriter(log_dir)

    metadata = load_metadata(args.data_path)
    seq_max_len = metadata['seq_max_len']
    args.segment_size = args.segment_size if args.segment_size > 1 else None
    args.layer_norm = True if args.layer_norm == 'True' else False
    dtype = None

    args.pad_masking = True if args.pad_masking=='True' else False

    print(f"# distance function: {args.distance}")
    
    # encoder model
    if args.encoder == 'trans':

        encoder = AttentionTransformerEncoder(d_model=args.enc_d_model, d_embedding=args.d_embedding, nhead=args.enc_nhead, 
                                              dim_feedforward=args.enc_d_ff, seq_max_len=seq_max_len, 
                                              dropout=args.dropout, activation=F.relu, enc_layers=args.n_enc_layers, 
                                              readout_layers=args.read_layers, segment_size=args.segment_size, 
                                              batch_first=True, norm_first=False, norm=args.layer_norm, 
                                              device=device, dtype=dtype)
        encoder._reset_parameters()
    
        args.result_dirname = f'trans_{args.distance}_nhead{args.enc_nhead}_dm{args.enc_d_model}_ff{args.enc_d_ff}_lyr{args.n_enc_layers}_emb{args.d_embedding}'

    elif args.encoder == 'cnn':
        encoder = CNNed(seq_len=seq_max_len, d_embedding=args.d_embedding, layers=args.n_enc_layers, channels=args.enc_d_model, 
                        kernel_size=3, stride=1, pooling='avg', readout_layers=args.read_layers, activation=True, 
                        batch_norm=True, device=device)

        args.result_dirname = f'cnn_{args.distance}_ch{args.enc_d_model}_lyr{args.n_enc_layers}_emb{args.d_embedding}'

    args.scaling = True if args.scaling == 'True' else False
    pair_encoder = PairEncoding(encoder, args.distance, args.scaling)
    pair_encoder.to(device)
    
    enc_args = to_enc_dict(args.enc_d_model, args.d_embedding, args.enc_nhead, args.enc_d_ff, args.dropout, F.relu, 
                          args.n_enc_layers, args.read_layers, args.segment_size, 
                          batch_first=True, norm_first=False, norm=args.layer_norm, 
                          distance=args.distance, scaling=args.scaling)
    
    # warm-up lambda function
    eps = 1e-6
    lr_formula = lambda epoch: min(args.lr_warmup_step**(-1) * (epoch+eps), args.lr_warmup_step**0.5 * (epoch+eps)**(-0.5))

    enc_params = count_parameters(pair_encoder)

    # decoder model
    if args.task == 'vanilla':
        print("# task: Vanilla Edit Distance Approximation")

        optimizer = optim.RAdam(pair_encoder.parameters(), lr=args.ed_lr, weight_decay=args.weight_decay)
        # learning rate warm-up scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_formula)

        decoder = None

        print(f"# num Encoder parameters: {enc_params}")

        dec_dirname = ''

    elif args.task == 'nw' or args.task == 'ednw':
        if args.task == 'ednw':
            print("# task: Needleman-Wunsch + Edit Distance Approximation")
        elif args.task == 'nw':
            print("# task: Needleman Wunsch embedding")

        decoder = MultiheadAttentionMapper(d_model=args.dec_d_model, d_embedding=args.d_embedding, nhead=args.dec_nhead, 
                                           seq_max_len=seq_max_len, d_ff=args.dec_d_ff, readin_layers=1, 
                                           sa_layers=args.n_dec_layers, dropout=args.dropout)
        decoder.to(device)

        dec_args = to_dec_dict(args.dec_d_model, args.d_embedding, args.dec_nhead, args.dec_d_ff, 1, args.n_dec_layers, args.dropout)

        optimizer = optim.RAdam([
            {'params': pair_encoder.parameters(), 'lr': args.ed_lr}, 
            {'params': decoder.parameters(), 'lr': args.nw_lr}, 
            ], weight_decay=args.weight_decay
        )

        # learning rate warm-up scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=[lr_formula, lr_formula])
     
        print(f"# num Encoder parameters: {enc_params}")
        dec_params = count_parameters(decoder)
        print(f"# num Decoder parameters: {dec_params}")

        dec_dirname = f'_nwa_nhead{args.dec_nhead}_dm{args.dec_d_model}_ff{args.dec_d_ff}_lyr{args.n_dec_layers}'

    normed_K = args.homologous_K / seq_max_len
    
    loss = LossMeter(args.task, args.mat_lambda, normed_K, args.ed_loss, args.nw_loss, 
                     args.loss_ed_smdw, args.loss_nw_smdw, args.loss_smdw_method, 
                     seq_max_len, device)
    
    # nw lambda for hyperparam descripted dirname
    nw_lam = '' if args.task == 'vanilla' else '_nw{:.0e}'.format(args.mat_lambda)
    args.result_dirname = args.result_dirname + dec_dirname + nw_lam + f'_{date}'
    print(args.result_dirname)

    checkpoints_dir = os.path.dirname(f'./model/params/{args.task}/{args.result_dirname}/')
    best = 1e15
    best_epoch = -1
    start_epoch = 1

    print("# Start Training")
    train_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        start_epoch_time = time.time()

        
        loss_total_train, loss_ed_train, loss_nw_train = train_epoch(args, dataset_list["train"], pair_encoder, 
                                                                     optimizer, loss, decoder, device)

        loss_total_val, loss_ed_val, loss_nw_val = evaluate(args, dataset_list["val"], pair_encoder, 
                                                            loss, decoder, device)
        scheduler.step()


        # tensorboard 
        writer.add_scalar('total_train_loss', loss_total_train, epoch) # train
        writer.add_scalar('ed_train_loss', loss_ed_train, epoch)
        writer.add_scalar('nw_train_loss', loss_nw_train, epoch)
        writer.add_scalar('lambda_scaled_train_nw', loss_nw_train*args.mat_lambda, epoch)

        writer.add_scalar('total_val_loss', loss_total_val, epoch) # val
        writer.add_scalar('ed_val_loss', loss_ed_val, epoch)
        writer.add_scalar('nw_val_loss', loss_nw_val, epoch)
        writer.add_scalar('lambda_scaled_val_nw', loss_nw_val*args.mat_lambda, epoch)
        
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"  epoch {epoch}")
            print(f"train loss | total: {loss_total_train} | encoder: {loss_ed_train} | decoder: {loss_nw_train} | {round(time.time()-start_epoch_time, 2)} seconds")
            print(f"val loss | total : {loss_total_val} | encoder: {loss_ed_val} | decoder: {loss_nw_val} | {round(time.time()-start_epoch_time, 2)} seconds") 
            sys.stdout.flush()
            
            # save checkpoints
            save_checkpoints(pair_encoder, enc_args, f'{checkpoints_dir}/encoder_ckpts_{epoch}.pt')
            if args.task == 'ednw':
                save_checkpoints(decoder, dec_args, f'{checkpoints_dir}/decoder_ckpts_{epoch}.pt')

        # save best model
        if loss_total_val < best:
            save_checkpoints(pair_encoder, enc_args, f'{checkpoints_dir}/encoder_best_{epoch}.pt')

            if args.task == 'ednw':
                save_checkpoints(decoder, dec_args, f'{checkpoints_dir}/decoder_best_{epoch}.pt')

            if best_epoch > 0:
                os.remove(f'{checkpoints_dir}/encoder_best_{best_epoch}.pt')

                if args.task == 'ednw':
                    os.remove(f'{checkpoints_dir}/decoder_best_{best_epoch}.pt')
            
            best = loss_total_val
            best_epoch = epoch

    # close SummaryWriter after training finished
    writer.close()

    print(f"# optimized after {args.epochs} epochs / {round(time.time()-train_time, 2)} seconds")
    print(f"# best epoch: {best_epoch}")

    # init model to load state_dict
    enc_ckpts = torch.load(f'{checkpoints_dir}/encoder_best_{best_epoch}.pt')
    enc_args = enc_ckpts['model_args']

    if args.encoder == 'trans':
        encoder = AttentionTransformerEncoder(d_model=enc_args['d_model'], d_embedding=enc_args['d_embedding'], nhead=enc_args['nhead'], 
                                              dim_feedforward=enc_args['dim_feedforward'], seq_max_len=seq_max_len, 
                                              dropout=enc_args['dropout'], activation=enc_args['activation'], 
                                              enc_layers=enc_args['enc_layers'], readout_layers=enc_args['readout_layers'], 
                                              segment_size=enc_args['segment_size'], 
                                              batch_first=enc_args['batch_first'], norm_first=enc_args['norm_first'], 
                                              norm=enc_args['norm'], 
                                              device=device, dtype=dtype)

    elif args.encoder == 'cnn':
        encoder = CNNed(seq_len=seq_max_len, d_embedding=enc_args['d_embedding'], layers=enc_args['enc_layers'], 
                        channels=enc_args['d_model'], kernel_size=3, stride=1, pooling='avg', 
                        readout_layers=enc_args['readout_layers'], activation=True, 
                        batch_norm=True, device=device)

    p_encoder = PairEncoding(encoder, enc_args['distance'], enc_args['scaling'])

    p_encoder.load_state_dict(enc_ckpts['model_state_dict'])
    p_encoder.to(device)
    
    if args.task in ['ednw']:
        dec_ckpts = torch.load(f'{checkpoints_dir}/decoder_best_{best_epoch}.pt')
        dec_args = dec_ckpts['model_args']

        decoder = MultiheadAttentionMapper(d_model=dec_args['d_model'], d_embedding=dec_args['d_embedding'], nhead=dec_args['nhead'], 
                                           seq_max_len=seq_max_len, d_ff=dec_args['dim_feedforward'], 
                                           readin_layers=dec_args['readin_layers'], sa_layers=dec_args['sa_layers'], dropout=dec_args['dropout'])
        
        decoder.load_state_dict(dec_ckpts['model_state_dict'])
        decoder.to(device)

    args.plot = True if args.plot == 'True' else False
    for key in dataset_list.keys():
        print(f"Plotting {key}_set edit distance...")
        
        inf_st = time.time()
        loss_total_test, loss_ed_test, loss_nw_test = test_and_plot(args, dataset_list[key], 
                                                                    p_encoder, loss, key=key, decoder=decoder, 
                                                                    plot=args.plot, norm_constant=seq_max_len, 
                                                                    device=device, date=date)
    
    print(f"test loss | total: {loss_total_test} | encoder: {loss_ed_test} | decoder: {loss_nw_test} | {round(time.time()-inf_st, 2)} seconds")

    args.plot_attn = True if args.plot_attn == 'True' else False
    if args.task in ['ednw'] and args.plot_attn:
        print("Visualizing Attention Map...")
        eval_attention(args, p_encoder, decoder, norm_constant=seq_max_len, idx=args.attn_idx, date=date, device=device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(args, dset_list, ed_model, optimizer, loss, decoder=None, device=None):
    
    ed_model.train()
    loss.train() # loss train mode

    if decoder is not None:
        decoder.train()

    loss.zero_sum()

    for fln in dset_list:
        loader = get_dataloader(load_dataset(fln, args.task),
                                args.batch_size, True, args.num_workers)

        for src, tgt, distance, nw_matrix in loader:
            src, tgt, distance = src.to(device), tgt.to(device), distance.to(device)

            if args.task == 'ednw':
                nw_matrix = nw_matrix.to(device)
        
            optimizer.zero_grad()

            if args.pad_masking:
                src_pad_mask = generate_pad_mask(src)
                tgt_pad_mask = generate_pad_mask(tgt)
            else:
                src_pad_mask = None
                tgt_pad_mask = None

            enc_src, enc_tgt, enc_distance = ed_model(src, tgt, src_pad_mask, tgt_pad_mask)

            if args.task == 'vanilla':
                nw_out = None
                nw_label = None
        
            elif args.task == 'nw' or args.task == 'ednw':
                nw_out = decoder(enc_src, enc_tgt)
                nw_label = nw_matrix
                
            loss.backprop(enc_distance, distance, nw_out, nw_label, src.shape[0])
            optimizer.step()
            
    total_loss, ed_loss, nw_loss = loss.get_avg()
            
    return total_loss, ed_loss, nw_loss


def evaluate(args, dset_list, ed_model, loss, decoder=None, device=None):
    ed_model.eval()
    loss.eval() # loss eval mode

    if decoder is not None:
        decoder.eval()

    loss.zero_sum()

    with torch.no_grad():
        for fln in dset_list:
            loader = get_dataloader(load_dataset(fln, args.task),
                                    args.batch_size, True, args.num_workers)

            for src, tgt, distance, nw_matrix in loader:
                src, tgt, distance = src.to(device), tgt.to(device), distance.to(device)
                
                if args.task == 'ednw':
                    nw_matrix = nw_matrix.to(device)

                if args.pad_masking:
                    src_pad_mask = generate_pad_mask(src)
                    tgt_pad_mask = generate_pad_mask(tgt)
                else:
                    src_pad_mask = None
                    tgt_pad_mask = None
            
                enc_src, enc_tgt, enc_distance = ed_model(src, tgt, src_pad_mask, tgt_pad_mask)
                
                if args.task == 'vanilla':
                    nw_out = None
                    nw_label = None
                elif args.task == 'nw' or args.task == 'ednw':
                    nw_out = decoder(enc_src, enc_tgt)
                    nw_label = nw_matrix

                loss.update(enc_distance, distance, nw_out, nw_label, src.shape[0])
            
    total_loss, ed_loss, nw_loss = loss.get_avg()

    return total_loss, ed_loss, nw_loss


def test_and_plot(args, dset_list, ed_model, loss, key, decoder=None, plot=True, norm_constant=1, 
                  device=None, date=None):
    
    ed_model.eval()
    loss.eval() # turn off regularization

    if decoder is not None:
        decoder.eval()
    
    loss.zero_sum()

    label_list = []
    output_list = []

    with torch.no_grad():
        for fln in tqdm(dset_list):
            loader = get_dataloader(load_dataset(fln, args.task),
                                    args.batch_size, True, args.num_workers)

            for src, tgt, distance, nw_matrix in loader:
                src, tgt, distance = src.to(device), tgt.to(device), distance.to(device)
                if args.task == 'ednw':
                    nw_matrix = nw_matrix.to(device)

                if args.pad_masking:
                    src_pad_mask = generate_pad_mask(src)
                    tgt_pad_mask = generate_pad_mask(tgt)
                else:
                    src_pad_mask = None
                    tgt_pad_mask = None

                enc_src, enc_tgt, enc_distance = ed_model(src, tgt, src_pad_mask, tgt_pad_mask)

                if args.task == 'vanilla':
                    nw_out = None
                    nw_label = None

                elif args.task == 'ednw':
                    nw_out = decoder(enc_src, enc_tgt)
                    nw_label = nw_matrix

                loss.update(enc_distance, distance, nw_out, nw_label, src.shape[0])

                label_list.append(distance.cpu().detach().numpy())
                output_list.append(enc_distance.cpu().detach().numpy())
    
    total_loss, ed_loss, nw_loss = loss.get_avg()

    outputs = np.concatenate(output_list, axis=0) * norm_constant
    labels = np.concatenate(label_list, axis=0) * norm_constant

    corr_coeff, _ = pearsonr(outputs, labels)
    err = rmse(outputs, labels, norm_constant)

    distance_dir = os.path.dirname(f'./distances/{args.task}/{args.result_dirname}/')
    if not os.path.exists(distance_dir):
        os.makedirs(distance_dir)

    ed_dir = f'{distance_dir}/{key}_{date}'
    ed_h_dir = f'{distance_dir}/{key}_K{args.homologous_K}_{date}'

    pickle.dump((labels, outputs), open(f"{ed_dir}.pkl", "wb"))

    if plot:
        plt.plot(labels, outputs, '.', color='blue', alpha=0.1, markersize=1)
        plt.plot(labels, labels, 'r-')
        plt.savefig(f'{ed_dir}.{args.format}', format=args.format)
        plt.close()
        
    
    if plot and args.homologous_K is not None:
        labels_h = labels[labels<args.homologous_K]
        outputs_h = outputs[labels<args.homologous_K]

        err_h = rmse(outputs_h, labels_h, norm_constant)
        corr_coeff_h, _ = pearsonr(outputs_h, labels_h)

        plt.plot(labels_h, outputs_h, '.', color='#4c4a73', alpha=0.2, markersize=1)
        plt.plot(labels_h, labels_h, 'r-')
        plt.title(f"K = {args.homologous_K}", fontsize=12)
        plt.savefig(f'{ed_h_dir}.{args.format}', format=args.format)
        plt.close()

    return total_loss, ed_loss, nw_loss

# visualize attention map 
def eval_attention(args, ed_model, decoder, norm_constant=1, idx=-1, date=None, device=None):
    ed_model.eval()
    decoder.eval()

    data_dir = f'{args.data_path}/attn.pkl'
    fig_dir = f'./distances/{args.task}/{args.result_dirname}/attention_{date}.{args.format}'
    
    dataset = load_dataset(data_dir, args.task)
    loader = get_dataloader(dataset, len(dataset), shuffle=False)
    if idx < 0:
        idx = random.randint(0, len(dataset)-1)
    
    with torch.no_grad():
        for src, tgt, d, m in loader:
            src, tgt, d, m = src.to(device), tgt.to(device), d.to(device), m.to(device)

            enc_src, enc_tgt, _ = ed_model(src, tgt)
            attn_weight = decoder(enc_src, enc_tgt)
    
    attn_weight = attn_weight[idx].cpu().detach().numpy()
    mat = m[idx].cpu().detach().numpy()

    err = rmse(attn_weight, mat)

    d = d[idx] * norm_constant
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), gridspec_kw={'width_ratios':[1, 1, 0.05]})
    plt.subplots_adjust(left=0.05,
                    bottom=0.0,
                    right=0.95,
                    top=1.0,
                    wspace=0.1,
                    hspace=0.01)
    
    # plt.title(f"ED = {round(d.item())}")
    # fig.text(0.5, 0.02, f"RMSE = {err}", ha='center')
    sns.heatmap(attn_weight, square=True, vmin=0.0, vmax=1.0, annot=False, ax=axes[0], cbar=False, xticklabels=False, yticklabels=False, cmap='rocket_r')
    sns.heatmap(mat, square=True, vmin=0.0, vmax=1.0, annot=False, ax=axes[1], cbar_ax=axes[2], xticklabels=False, yticklabels=False, cmap='rocket_r')
    axes[2].set_position([0.93, 0.2, 0.02, 0.6])
    plt.savefig(fig_dir, format=args.format)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vanilla', help='Task :: "vanilla", "nw", "ednw" ')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    parser.add_argument('--seed', type=int, default=404, help='Random Seed')
    parser.add_argument('--data_path', type=str, default='./data/qiita', help='Dataset path')
    parser.add_argument('--encoder', type=str, default='trans', help='encoder type')
    parser.add_argument('--distance', type=str, default='hyperbolic', help='Distance function type')
    parser.add_argument('--scaling', type=str, default=False, help='Project to hypersphere (for hyperbolic')
    parser.add_argument('--enc_nhead', type=int, default=2, help='Trans encoder heads')
    parser.add_argument('--dec_nhead', type=int, default=4, help='NWA nheads')
    parser.add_argument('--enc_d_model', type=int, default=16, help='encoder hidden dimension')
    parser.add_argument('--dec_d_model', type=int, default=64, help='decoder hidden dimension')
    parser.add_argument('--d_embedding', type=int, default=128, help='dimension which to embed sequences')
    parser.add_argument('--enc_d_ff', type=int, default=16, help='Transformer encoder feedforward dimension')
    parser.add_argument('--dec_d_ff', type=int, default=1024, help='NW attention feedforward dimension')
    parser.add_argument('--segment_size', type=int, default=1, help='segment_size')
    parser.add_argument('--n_enc_layers', type=int, default=2, help='Number of Encoder layers')
    parser.add_argument('--n_dec_layers', type=int, default=2, help='Number of Decoder layers')
    parser.add_argument('--read_layers', type=int, default=1, help='Number of readout/in layers')
    parser.add_argument('--layer_norm', type=str, default='True', help='Boolean for layer norm')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--ed_loss', type=str, default='mse', help='ed loss')
    parser.add_argument('--nw_loss', type=str, default='jsd', help='nw loss')
    parser.add_argument('--loss_ed_smdw', type=str, default='False', help='ed Loss sm distance weight')
    parser.add_argument('--loss_nw_smdw', type=str, default='False', help='nw Loss sm distance weight')
    parser.add_argument('--loss_smdw_method', type=str, default='sig', help='sm distance weighting method')
    parser.add_argument('--mat_lambda', type=float, default=1.0, help='Scaling Factor for NW loss')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--ed_lr', type=float, default=0.01, help='encoder Learning rate')
    parser.add_argument('--nw_lr', type=float, default=0.001, help='decoder(nw) Learning rate')
    parser.add_argument('--lr_warmup_step', type=int, default=20, help='LR scheduler step size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--pad_masking', type=str, default='False', help='pad masking')
    parser.add_argument('--homologous_K', type=int, default=None, help='threshold K for homologous sequences')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--print_every', type=int, default=5, help='Print training results every')
    parser.add_argument('--plot', type=str, default='True', help='Plot real X predicted distances')
    parser.add_argument('--plot_attn', type=str, default='True', help='Plot attention map')
    parser.add_argument('--format', type=str, default='png', help='Plot figure format')
    parser.add_argument('--attn_idx', type=int, default=-1, help='seq pair idx for attn plot')
    args = parser.parse_args()
    
    if args.task not in ['vanilla', 'ednw']:
        raise ValueError(f'{args.task} Invalid Task')

    train(args)

