# main.py
import time
import threading
import pickle
import wandb
import numpy as np
from collections import deque
from services.buffer_client import BufferClient
from services.rollout_client import RolloutClient
from trainers.trainer import Trainer

class MainLoop:
    def __init__(self,
                 buffer_address='localhost:50052',
                 rollout_address='localhost:50051',
                 min_buffer_size=1000,
                 batch_size=32):
        # 初始化wandb
        wandb.init(
            project="titan-rl",
            config={
                "env": "CartPole-v1",
                "algorithm": "DQN",
                "min_buffer_size": min_buffer_size,
                "batch_size": batch_size,
                "gamma": 0.99,
                "lr": 0.001
            }
        )
        
        self.buffer_client = BufferClient(buffer_address)
        self.rollout_client = RolloutClient(rollout_address)
        self.trainer = Trainer(state_dim=4, action_dim=2)
        
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.running = False
        
        # 训练统计
        self.total_steps = 0
        self.total_episodes = 0
        self.training_steps = 0
        
        # 记录最近100个episode的reward
        self.episode_rewards = deque(maxlen=100)
        self.current_episode_reward = 0
        
    def _collect_experience(self):
        """经验收集线程"""
        while self.running:
            policy_weights = self.trainer.get_policy_weights()
            
            experiences, metrics = self.rollout_client.collect_experience(
                policy_weights=policy_weights,
                num_steps=1000  # 确保这里是整数
            )
            
            # 更新统计信息
            self.total_steps += int(metrics['steps'])  # 确保转换为整数
            
            # 处理每条经验，累积reward
            for exp in experiences:
                self.current_episode_reward += exp['reward']
                if exp['done']:
                    self.total_episodes += 1
                    self.episode_rewards.append(self.current_episode_reward)
                    
                    # 记录episode相关指标
                    wandb.log({
                        "episode": self.total_episodes,
                        "episode_reward": self.current_episode_reward,
                        "episode_length": int(metrics['steps']),  # 确保转换为整数
                        "mean_100_reward": np.mean(self.episode_rewards),
                    }, step=self.total_steps)
                    
                    self.current_episode_reward = 0
            
            self.buffer_client.add_experience(experiences)
            
            # 记录收集相关的指标
            wandb.log({
                "buffer_size": self.buffer_client.get_status().current_size,
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
            }, step=self.total_steps)
            
            # 如果buffer较满则等待
            status = self.buffer_client.get_status()
            if status.current_size > self.min_buffer_size * 2:
                time.sleep(1)
                
    def _train(self):
        """训练线程"""
        while self.running:
            status = self.buffer_client.get_status()
            if status.current_size < self.min_buffer_size:
                print(f"Buffer size ({status.current_size}) < min size ({self.min_buffer_size})")
                time.sleep(1)
                continue
                
            batch = self.buffer_client.sample_batch(self.batch_size)
            metrics = self.trainer.train_step(batch)
            self.training_steps += 1
            
            # 记录训练相关的指标
            wandb.log({
                "train/loss": float(metrics['loss']),  # 确保是Python float
                "train/avg_q_value": float(metrics['avg_q_value']),
                "train/max_q_value": float(metrics['max_q_value']),
                "train/min_q_value": float(metrics['min_q_value']),
                "train/avg_target": float(metrics['avg_target']),
                "training_steps": self.training_steps,
            }, step=self.total_steps)
            
            # 每100步打印一次状态
            if self.training_steps % 100 == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                print(f"\nTraining step: {self.training_steps}")
                print(f"Total environment steps: {self.total_steps}")
                print(f"Episodes: {self.total_episodes}")
                print(f"Mean 100 episode reward: {mean_reward:.2f}")
                print(f"Loss: {float(metrics['loss']):.4f}")
                print(f"Average Q-value: {float(metrics['avg_q_value']):.4f}")
                
                # 保存检查点
                self.trainer.save_checkpoint('checkpoint.pth')
                
    def run(self, total_steps=1000000):
        """运行训练循环"""
        self.running = True
        print("Starting training...")
        
        collect_thread = threading.Thread(target=self._collect_experience)
        collect_thread.start()
        
        train_thread = threading.Thread(target=self._train)
        train_thread.start()
        
        try:
            while self.total_steps < total_steps and self.running:
                time.sleep(1)
                
                # 检查是否已经解决了环境
                if len(self.episode_rewards) >= 100:
                    mean_reward = np.mean(self.episode_rewards)
                    if mean_reward >= 495:  # CartPole-v1 的解决标准
                        print(f"\nEnvironment solved in {self.total_episodes} episodes!")
                        print(f"Mean reward over last 100 episodes: {mean_reward:.2f}")
                        self.running = False
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False
            
        collect_thread.join()
        train_thread.join()
        
        # 保存最终模型
        self.trainer.save_checkpoint('final_model.pth')
        wandb.finish()

def main():
    loop = MainLoop()
    loop.run()

if __name__ == '__main__':
    main()