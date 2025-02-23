# rollout_client.py
import grpc
import torch
import pickle
from proto import rollout_pb2
from proto import rollout_pb2_grpc
from models.simple_policy import SimplePolicy

class RolloutClient:
    def __init__(self, address='localhost:50051'):
        self.channel = grpc.insecure_channel(address)
        self.stub = rollout_pb2_grpc.RolloutServiceStub(self.channel)
    
    def collect_experience(self, policy_weights=None, num_steps=100):
        # 准备请求
        request = rollout_pb2.CollectRequest(
            policy_weights=pickle.dumps(policy_weights) if policy_weights else b'',
            num_steps=num_steps,
            collection_id='test'
        )
        
        # 发送请求
        response = self.stub.CollectExperience(request)
        
        # 解析返回的经验数据
        experiences = pickle.loads(response.experiences)
        
        return experiences, response.metrics

def main():
    # 创建策略网络
    policy = SimplePolicy(4, 2)  # CartPole环境: 4维状态, 2个动作
    
    # 创建客户端
    client = RolloutClient()
    
    # 收集一些经验
    experiences, metrics = client.collect_experience(
        policy_weights=policy.state_dict(),
        num_steps=200
    )
    
    print(f"Collected {len(experiences)} experiences")
    print(f"Metrics: {metrics}")
    
    # 打印第一条经验数据
    print("\nFirst experience:")
    print(experiences[0])

if __name__ == '__main__':
    main()