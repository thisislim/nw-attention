import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn

from model.transformer import AttentionTransformerEncoder
from model.mha import MultiheadAttentionMapper
from model.pair_encoding import PairEncoding
from model.cnn import CNNed
from data_utils.data_handler import load_dataset, get_dataloader, load_metadata, load_dataset_list
from utils.math_and_loss import LossMeter
from train import test_and_plot, eval_attention



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vanilla', help='Task :: "vanilla", "nw", "ednw" ')
    parser.add_argument('--seed', type=int, default=404, help='Random Seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    parser.add_argument('--data_path', type=str, default='./data/qiita', help='Dataset path')
    parser.add_argument('--encoder', type=str, default='trans', help='encoder type')
    parser.add_argument('--encoder_path', type=str, default='', help='Encoder model path')
    parser.add_argument('--decoder_path', type=str, default='', help='decoder path')
    parser.add_argument('--ed_loss', type=str, default='mse', help='ed loss')
    parser.add_argument('--nw_loss', type=str, default='jsd', help='nw loss')
    parser.add_argument('--pad_masking', type=str, default='False', help='pad masking')
    parser.add_argument('--mat_lambda', type=float, default=1.0, help='Scaling Factor for NW loss')  
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--distance', type=str, default='hyperbolic', help='Distance function type')
    parser.add_argument('--homologous_K', type=int, default=None, help='threshold K for homologous sequences')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--plot', default=False, help='Plot real X predicted distances')
    parser.add_argument('--plot_attn', type=str, default='True', help='Plot attention map')
    parser.add_argument('--format', default='png', help='Plot figure format')
    parser.add_argument('--attn_idx', type=int, default=-1, help='seq pair idx for attn plot')
    parser.add_argument('--test_only', type=str, default='False', help='run for test set only')
    args = parser.parse_args()
    
    if args.task not in ['vanilla', 'ednw']:
        raise ValueError(f'{args.task} Invalid Task')
    
    if torch.cuda.is_available() and args.cuda < torch.cuda.device_count():
        device = f'cuda:{args.cuda}'
    else:
        device = 'cpu'
    device = torch.device(device)
    print("# device: '{}'".format(device))

    torch.manual_seed(args.seed)

    args.test_only = True if args.test_only == 'True' else False
    args.pad_masking = True if args.pad_masking == 'True' else False

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.result_dirname = os.path.dirname(args.encoder_path).split('/')[-1]
    print(args.result_dirname)

    metadata = load_metadata(args.data_path)
    seq_max_len = metadata['seq_max_len']
    
    dataset_list = load_dataset_list(args.data_path)

    normed_K = args.homologous_K / seq_max_len

    loss = LossMeter(args.task, args.mat_lambda, normed_K, args.ed_loss, args.nw_loss, device=device)

    enc_ckpts = torch.load(args.encoder_path)
    enc_args = enc_ckpts['model_args']

    if args.encoder == 'trans':
        encoder = AttentionTransformerEncoder(d_model=enc_args['d_model'], d_embedding=enc_args['d_embedding'], nhead=enc_args['nhead'], 
                                              dim_feedforward=enc_args['dim_feedforward'], seq_max_len=seq_max_len, 
                                              dropout=enc_args['dropout'], activation=enc_args['activation'], 
                                              enc_layers=enc_args['enc_layers'], readout_layers=enc_args['readout_layers'], 
                                              segment_size=enc_args['segment_size'], 
                                              batch_first=enc_args['batch_first'], norm_first=enc_args['norm_first'], 
                                              norm=enc_args['norm'], 
                                              device=device)
    elif args.encoder == 'cnn':
        encoder = CNNed(seq_len=seq_max_len, d_embedding=enc_args['d_embedding'], layers=enc_args['enc_layers'], 
                        channels=enc_args['d_model'], kernel_size=3, stride=1, pooling='avg', 
                        readout_layers=enc_args['readout_layers'], activation=True, 
                        batch_norm=True, device=device)
                        
    p_encoder = PairEncoding(encoder, enc_args['distance'], enc_args['scaling'])

    p_encoder.load_state_dict(enc_ckpts['model_state_dict'])
    p_encoder.to(device)

    decoder = None
    

    if args.task in ['ednw']:
        dec_ckpts = torch.load(args.decoder_path)
        dec_args = dec_ckpts['model_args']

        decoder = MultiheadAttentionMapper(d_model=dec_args['d_model'], d_embedding=dec_args['d_embedding'], nhead=dec_args['nhead'], 
                                           seq_max_len=seq_max_len, d_ff=dec_args['dim_feedforward'], 
                                           readin_layers=dec_args['readin_layers'], sa_layers=dec_args['sa_layers'], dropout=dec_args['dropout'])
        
        decoder.load_state_dict(dec_ckpts['model_state_dict'])
        decoder.to(device) 

    if args.test_only:
        print("Plotting test_set edit distances...")
        norm_constant = metadata['seq_max_len']
        loss_total_test, loss_ed_test, loss_nw_test = test_and_plot(args, dataset_list['test'], 
                                                                    p_encoder, loss, key='test', decoder=decoder, 
                                                                    plot=args.plot, norm_constant=seq_max_len, 
                                                                    device=device, date=date)        
        
    else:
        for key in dataset_list.keys():
            print(f"Plotting {key}_set edit distance...")
            norm_constant = metadata['seq_max_len']
            loss_total_test, loss_ed_test, loss_nw_test = test_and_plot(args, dataset_list[key], 
                                                                    p_encoder, loss, key=key, decoder=decoder, 
                                                                    plot=args.plot, norm_constant=seq_max_len, 
                                                                    device=device, date=date)

    print(f"test loss | total: {loss_total_test} | encoder: {loss_ed_test} | decoder: {loss_nw_test}")

    args.plot_attn = True if args.plot_attn == 'True' else False
    if args.task in ['ednw'] and args.plot_attn:
        print("Visualizing Attention Map...")
        eval_attention(args, p_encoder, decoder, norm_constant=seq_max_len, idx=args.attn_idx, date=date, device=device)

