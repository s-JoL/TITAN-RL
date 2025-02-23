import yaml
import grpc
import pickle
import argparse
import importlib
import gymnasium as gym
from concurrent import futures
from proto import rollout_pb2
from proto import rollout_pb2_grpc

class RolloutServicer(rollout_pb2_grpc.RolloutServiceServicer):
    def __init__(self, config_path='config/dqn.yaml'):
        # 加载配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.num_workers = config['rollout_server']['num_workers']
        self.envs = [gym.make(config['rollout_server']['env_name']) for _ in range(self.num_workers)]
        
        # 从配置中获取策略信息
        policy_path = config['policy']['path']
        policy_kwargs = config['policy']['kwargs']
        
        # 从字符串路径导入策略类
        module_name, class_name = policy_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        policy_class = getattr(module, class_name)
        self.policy = policy_class(**policy_kwargs)
        self.total_steps = 0
        
    def CollectExperience(self, request, context):
        # 更新策略权重
        if request.policy_weights:
            self.policy.load_weights(request.policy_weights)
            
        experiences = []
        total_reward = 0
        steps_this_collect = 0
        
        # 使用第一个环境收集数据(这里简化为单个环境)
        env = self.envs[0]
        state, _ = env.reset()
        
        for _ in range(request.num_steps):
            # 选择动作
            action = self.policy.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录经验
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }
            experiences.append(experience)
            
            total_reward += reward
            steps_this_collect += 1
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
                
        self.total_steps += steps_this_collect
        
        # 序列化经验数据
        serialized_experiences = pickle.dumps(experiences)
        
        # 准备指标
        metrics = {
            'mean_reward': total_reward / steps_this_collect,
            'steps': steps_this_collect
        }
        
        return rollout_pb2.CollectResponse(
            experiences=serialized_experiences,
            collection_id=request.collection_id,
            metrics=metrics
        )
    
    def GetStatus(self, request, context):
        return rollout_pb2.StatusResponse(
            num_workers=self.num_workers,
            total_steps=self.total_steps,
            status='running'
        )

def serve():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Rollout Server')
    parser.add_argument('--config', 
                       type=str,
                       default='config/dqn.yaml',
                       help='Path to configuration file')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rollout_pb2_grpc.add_RolloutServiceServicer_to_server(
        RolloutServicer(config_path=args.config), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()