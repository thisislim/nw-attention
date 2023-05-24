import torch
import torch.nn as nn
from torch.nn.init import normal_


class FCLayer(nn.Module):

    def __init__(self, dim_in ,dim_out ,activation=True, dropout=0.0, b_norm=False, bias=True, device=None):
        super(FCLayer, self).__init__()

        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.dropout=nn.Dropout(p=dropout)
        self.b_norm = None
        if b_norm == True:
            self.b_norm = nn.BatchNorm1d(dim_out)
        self.activation = activation
        self.activation_fn = nn.ReLU()
        self.bias = bias

        self._reset_parameters()


    def _reset_parameters(self):
        normal_(self.linear.weight)
        if self.bias:
            self.linear.bias.data.zero_()
    

    def forward(self, x):
        
        output = self.linear(x)
        if self.activation is not None:
            output = self.activation_fn(output)
        output = self.dropout(output)
        if self.b_norm is not None:
            output = self.b_norm(output)

        return output



class MLP(nn.Module):

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, mid_activation=True, last_activation=False, 
                  dropout=0.0, mid_b_norm=True, last_b_norm=True, device=None):
        super(MLP, self).__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.FC = nn.ModuleList()

        if num_layers <= 1:
            self.FC.append(FCLayer(dim_in, dim_out, activation=last_activation, b_norm=last_b_norm, 
                                   dropout=dropout, device=device))

        else:
            self.FC.append(FCLayer(dim_in, dim_hidden, activation=last_activation, b_norm=last_b_norm, 
                                   dropout=dropout, device=device))
            for _ in range(num_layers - 2):
                self.FC.append(FCLayer(dim_hidden, dim_hidden, activation=mid_activation, b_norm=mid_b_norm, 
                                       dropout=dropout, device=device))
            self.FC.append(FCLayer(dim_hidden, dim_out, activation=last_activation, b_norm=last_b_norm, 
                                   dropout=dropout, device=device))


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        for fc in self.FC:
            x = fc(x)            
        return x

    def _reset_parameters(self):
        for fc in self.FC:
            fc._reset_parameters()
        
        
