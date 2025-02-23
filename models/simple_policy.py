import torch
import pickle
import random
from torch import nn

class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.to(self.device)
    
    def load_weights(self, weights):
        with torch.no_grad():
            state_dict = pickle.loads(weights)
            self.load_state_dict(state_dict)
    
    def act(self, state, eps=0.):
        if random.random() < eps:
            return random.randint(0, 1)
        with torch.no_grad():
            state = torch.FloatTensor(state)
            logits = self.forward(state)
            action = torch.argmax(logits).item()
            return action
        
    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)