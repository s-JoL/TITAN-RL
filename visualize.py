# visualize.py
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
from models.simple_policy import SimplePolicy
import os

def visualize_agent(model_path, video_folder='videos', num_episodes=5):
    # 创建视频保存目录
    os.makedirs(video_folder, exist_ok=True)
    
    # 创建环境并包装视频录制器
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda x: True  # 录制所有episode
    )
    
    # 创建策略网络并加载权重
    policy = SimplePolicy(state_dim=4, action_dim=2)
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 选择动作
            with torch.no_grad():
                action = policy.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                break
                
            state = next_state
    
    env.close()
    
    # 打印统计信息
    print("\nEvaluation Results:")
    print(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Average Episode Length: {sum(episode_lengths) / len(episode_lengths):.2f}")
    print(f"Max Reward: {max(episode_rewards)}")
    print(f"Max Episode Length: {max(episode_lengths)}")
    print(f"\nVideos saved to: {os.path.abspath(video_folder)}")

if __name__ == '__main__':
    # 使用训练好的模型
    model_path = 'checkpoint.pth'  # 或 'final_model.pth'
    visualize_agent(
        model_path=model_path,
        video_folder='videos/cartpole_evaluation',
        num_episodes=5
    )