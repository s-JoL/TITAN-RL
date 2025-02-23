# buffer_client.py
import grpc
import pickle
import buffer_pb2
import buffer_pb2_grpc

class BufferClient:
    def __init__(self, address='localhost:50052'):
        self.channel = grpc.insecure_channel(address)
        self.stub = buffer_pb2_grpc.BufferServiceStub(self.channel)
        
    def add_experience(self, experiences):
        """添加经验数据到buffer
        
        Args:
            experiences: 单条或多条经验数据
        """
        request = buffer_pb2.AddRequest(
            experiences=pickle.dumps(experiences),
            source_id='test'
        )
        return self.stub.AddExperience(request)
        
    def sample_batch(self, batch_size):
        """采样一个批次的数据
        
        Args:
            batch_size: 批次大小
        Returns:
            batch_data: 采样的数据批次
        """
        request = buffer_pb2.SampleRequest(batch_size=batch_size)
        response = self.stub.SampleBatch(request)
        return pickle.loads(response.batch_data)
    
    def get_status(self):
        """获取buffer状态"""
        request = buffer_pb2.StatusRequest()
        return self.stub.GetStatus(request)

def main():
    # 测试代码
    client = BufferClient()
    
    # 添加一些测试数据
    test_experiences = [
        {'state': [1, 2, 3, 4], 'action': 1, 'reward': 1.0},
        {'state': [2, 3, 4, 5], 'action': 0, 'reward': -1.0}
    ]
    
    response = client.add_experience(test_experiences)
    print("Add response:", response)
    
    # 采样数据
    batch = client.sample_batch(1)
    print("\nSampled batch:", batch)
    
    # 获取状态
    status = client.get_status()
    print("\nBuffer status:", status)

if __name__ == '__main__':
    main()