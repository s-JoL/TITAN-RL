# buffer_server.py
import grpc
import numpy as np
import pickle
from concurrent import futures
import buffer_pb2
import buffer_pb2_grpc
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, experience):
        """添加单条或多条经验"""
        if isinstance(experience, list):
            self.buffer.extend(experience)
        else:
            self.buffer.append(experience)
            
    def sample(self, batch_size):
        """随机采样一个批次"""
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]
    
    @property
    def size(self):
        return len(self.buffer)

class BufferServicer(buffer_pb2_grpc.BufferServiceServicer):
    def __init__(self, capacity=100000):
        self.buffer = ReplayBuffer(capacity)
        
    def AddExperience(self, request, context):
        try:
            experiences = pickle.loads(request.experiences)
            self.buffer.add(experiences)
            return buffer_pb2.AddResponse(
                success=True,
                message=f"Added {len(experiences)} experiences"
            )
        except Exception as e:
            return buffer_pb2.AddResponse(
                success=False,
                message=str(e)
            )
    
    def SampleBatch(self, request, context):
        if self.buffer.size < request.batch_size:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                         "Not enough samples in buffer")
            
        batch = self.buffer.sample(request.batch_size)
        serialized_batch = pickle.dumps(batch)
        return buffer_pb2.SampleResponse(batch_data=serialized_batch)
    
    def GetStatus(self, request, context):
        return buffer_pb2.StatusResponse(
            current_size=self.buffer.size,
            capacity=self.buffer.capacity,
            status='running'
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    buffer_pb2_grpc.add_BufferServiceServicer_to_server(
        BufferServicer(), server)
    server.add_insecure_port('[::]:50052')  # 使用不同的端口
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()