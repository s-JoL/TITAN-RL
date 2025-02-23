# trainers/ppo_trainer.py
import yaml
import torch
import importlib
from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self, config_path='config/ppo.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get policy info from config
        policy_path = config['policy']['path']
        policy_kwargs = config['policy']['kwargs']
        value_path = config['value_network']['path']
        value_kwargs = config['value_network']['kwargs']
        
        # Import policy and value network classes
        module_name, class_name = policy_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        policy_class = getattr(module, class_name)
        self.policy = policy_class(**policy_kwargs)
        
        module_name, class_name = value_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        value_class = getattr(module, class_name)
        self.value_net = value_class(**value_kwargs)
        
        # Training parameters
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['train']['policy_lr'])
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config['train']['value_lr'])
        self.clip_ratio = config['train']['clip_ratio']
        self.gamma = config['train']['gamma']
        self.value_loss_coef = config['train']['value_loss_coef']
        self.entropy_coef = config['train']['entropy_coef']
        
    def train_step(self, batch):
        """Execute one training step
        
        Args:
            batch: List of experiences with trajectory information
                {'state': array, 'action': int, 'reward': float,
                'next_state': array, 'done': bool, 'future_reward': float}
        """
        # Prepare training data
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        future_rewards = torch.FloatTensor([exp['future_reward'] for exp in batch]).to(self.device)
        
        # Get current policy distribution and values
        with torch.no_grad():
            action_probs = self.policy(states)
            curr_values = self.value_net(states).squeeze()
            
            # Calculate advantages using future_rewards - value estimates
            advantages = future_rewards - curr_values
            
            # Get log probabilities of actions
            dist = Categorical(action_probs)
            old_log_probs = dist.log_prob(actions)
        
        # PPO policy loss
        # Get new policy distribution
        new_action_probs = self.policy(states)
        new_dist = Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)
        
        # Calculate ratio and clipped ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        
        # Policy loss
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Entropy bonus
        entropy = new_dist.entropy().mean()
        policy_loss = policy_loss - self.entropy_coef * entropy
        
        # Value loss
        value_pred = self.value_net(states).squeeze()
        value_loss = self.value_loss_coef * torch.nn.functional.mse_loss(value_pred, future_rewards)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Return metrics
        return {
            'train/policy_loss': policy_loss.item(),
            'train/value_loss': value_loss.item(),
            'train/entropy': entropy.item(),
            'train/mean_advantage': advantages.mean().item(),
            'train/mean_ratio': ratio.mean().item(),
            'train/mean_value': curr_values.mean().item()
        }
    
    def get_policy_weights(self):
        """Get current policy weights"""
        return self.policy.state_dict()
    
    def save_checkpoint(self, path):
        """Save checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])