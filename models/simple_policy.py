import torch
import pickle

class SimplePolicy:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim)
        ).to(self.device)
    
    def load_weights(self, weights):
        with torch.no_grad():
            state_dict = pickle.loads(weights)
            self.net.load_state_dict(state_dict)
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits = self.net(state)
            action = torch.argmax(logits).item()
            return action