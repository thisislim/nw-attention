import os
import torch


def save_checkpoints(model, model_args, ckpts_path):
    ckpts_dir = os.path.dirname(ckpts_path)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_args': model_args}, ckpts_path)

def to_enc_dict(d_model, d_embedding, nhead, dim_feedforward, dropout, activation, enc_layers,readout_layers, 
                segment_size,batch_first, norm_first, norm, distance, scaling):
    
    enc_dict = {
        'd_model':d_model, 
        'd_embedding': d_embedding,
        'nhead': nhead, 
        'dim_feedforward': dim_feedforward, 
        'dropout': dropout, 
        'activation': activation, 
        'enc_layers': enc_layers, 
        'readout_layers': readout_layers, 
        'segment_size': segment_size, 
        'batch_first': batch_first, 
        'norm_first': norm_first, 
        'norm': norm, 
        'distance': distance, 
        'scaling': scaling
    }
    return enc_dict

def to_dec_dict(d_model, d_embedding, nhead, d_ff, readin_layers, sa_layers, dropout):
    dec_dict = {
        'd_model': d_model, 
        'd_embedding': d_embedding, 
        'nhead': nhead, 
        'dim_feedforward': d_ff, 
        'readin_layers': readin_layers, 
        'sa_layers': sa_layers, 
        'dropout': dropout, 
    }
    return dec_dict

def to_cnn_dict(channels, kernel_size, stride, layers, pooling, readout_layers, dropout):
    cnn_dict = {
        'channels': channels, 
        'kernel_size': kernel_size, 
        'stride': stride, 
        'layers': layers, 
        'pooling': pooling, 
        'readout_layers': readout_layers, 
        'dropout': dropout
    }
    return cnn_dict

    