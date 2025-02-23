# trainer.py
import torch
import numpy as np
from models import SimplePolicy

class Trainer:
    def __init__(self, state_dim=4, action_dim=2, lr=0.001, gamma=0.99):
        self.policy = SimplePolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.net.parameters(), lr=lr)
        self.gamma = gamma
        
    def train_step(self, batch):
        """执行一步训练
        
        Args:
            batch: 包含多条经验的列表，每条经验是一个字典
                  {'state': array, 'action': int, 'reward': float, 
                   'next_state': array, 'done': bool}
        
        Returns:
            dict: 训练相关的指标
        """
        # 准备训练数据
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.FloatTensor([exp['done'] for exp in batch])
        
        # 计算TD目标
        with torch.no_grad():
            next_q_values = self.policy.net(next_states).max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_values
            
        # 计算当前Q值
        q_values = self.policy.net(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算损失并更新
        loss = torch.nn.functional.mse_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 返回训练指标
        return {
            'loss': loss.item(),
            'avg_q_value': q_values.mean().item(),
            'avg_target': targets.mean().item(),
            'max_q_value': q_values.max().item(),
            'min_q_value': q_values.min().item(),
        }
    
    def get_policy_weights(self):
        """获取当前策略的权重"""
        return self.policy.net.state_dict()
    
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'policy_state_dict': self.policy.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.policy.net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])