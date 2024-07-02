import torch
import torch.nn as nn

activation_set = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leakyrelu": nn.LeakyReLU(),
    "prelu": nn.PReLU()
}

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation_func="prelu"):
        super(DNN, self).__init__()
        layers = [input_dim] + hidden_dims + [output_dim]
        self.net = []
        
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 1 and activation_func in activation_set:
                self.net.append(activation_set[activation_func])
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, input):
        return self.net(input)


class PositionwiseFFNN(nn.Module):
    """http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(PositionwiseFFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    """http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.p1 = nn.Parameter(torch.ones(features))
        self.p2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)

        return self.p1 * (input - mean) / (std + self.eps) + self.p2
