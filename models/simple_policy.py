# models.py
import torch
import pickle

class SimplePolicy:
    def __init__(self, state_dim, action_dim):
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim)
        )
    
    def load_weights(self, weights):
        with torch.no_grad():
            state_dict = pickle.loads(weights)
            self.net.load_state_dict(state_dict)
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            logits = self.net(state)
            action = torch.argmax(logits).item()
            return action