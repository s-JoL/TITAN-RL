import grpc
import gymnasium as gym
import pickle
from concurrent import futures
from proto import rollout_pb2
from proto import rollout_pb2_grpc

from models.simple_policy import SimplePolicy

class RolloutServicer(rollout_pb2_grpc.RolloutServiceServicer):
    def __init__(self, num_workers=1):
        self.num_workers = num_workers
        self.envs = [gym.make('CartPole-v1') for _ in range(num_workers)]
        self.policy = SimplePolicy(4, 2)  # CartPole: 4维状态, 2个动作
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rollout_pb2_grpc.add_RolloutServiceServicer_to_server(
        RolloutServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()