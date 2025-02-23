import torch
import pickle
from torch import nn
from torch.distributions import Categorical

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.to(self.device)
    
    def load_weights(self, weights):
        with torch.no_grad():
            state_dict = pickle.loads(weights)
            self.load_state_dict(state_dict)
    
    def act(self, state, *args):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs = self.forward(state)
            # 创建分类分布
            dist = Categorical(action_probs)
            # 采样动作
            action = dist.sample()
            # 返回动作
            return action.item()
        
    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)