# trainer.py
import yaml
import torch
import importlib

class DQNTrainer:
    def __init__(self, config_path='config/dqn.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # 从配置中获取策略信息
        policy_path = config['policy']['path']
        policy_kwargs = config['policy']['kwargs']
        
        # 从字符串路径导入策略类
        module_name, class_name = policy_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        policy_class = getattr(module, class_name)
        self.policy = policy_class(**policy_kwargs)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['train']['lr'])
        self.gamma = config['train']['gamma']
        
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
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
        
        # 计算TD目标
        with torch.no_grad():
            next_q_values = self.policy(next_states).max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_values
            
        # 计算当前Q值
        q_values = self.policy(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算损失并更新
        loss = torch.nn.functional.mse_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 返回训练指标
        return {
            'train/loss': loss.item(),
            'train/avg_q_value': q_values.mean().item(),
            'train/avg_target': targets.mean().item(),
            'train/max_q_value': q_values.max().item(),
            'train/min_q_value': q_values.min().item(),
        }
    
    def get_policy_weights(self):
        """获取当前策略的权重"""
        return self.policy.state_dict()
    
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])