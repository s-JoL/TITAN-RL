import torch
import pickle
from torch import nn

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 价值网络结构
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.to(self.device)
    
    def load_weights(self, weights):
        with torch.no_grad():
            state_dict = pickle.loads(weights)
            self.load_state_dict(state_dict)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = x.to(self.device)
        return self.net(x)
