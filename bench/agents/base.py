from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..dataclasses import SEDTask, SearchResult, Equation

@dataclass
class AgentMessage:
    """智能体间通信的消息格式"""
    sender: str
    receiver: str
    message_type: str  # "theory_hypothesis", "experiment_result", "feedback", "coordination"
    content: Dict[str, Any]
    timestamp: float
    message_id: str

@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    current_task: Optional[SEDTask]
    working_memory: Dict[str, Any]
    hypotheses: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]

class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.state = AgentState(
            agent_id=agent_id,
            current_task=None,
            working_memory={},
            hypotheses=[],
            confidence_scores={}
        )
        self.message_queue = []
        self.communication_log = []
    
    @abstractmethod
    def process_task(self, task: SEDTask) -> List[SearchResult]:
        """处理任务的核心方法"""
        pass
    
    @abstractmethod
    def generate_hypothesis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成假设"""
        pass
    
    @abstractmethod
    def evaluate_hypothesis(self, hypothesis: Dict[str, Any], data: Dict[str, Any]) -> float:
        """评估假设"""
        pass
    
    def send_message(self, receiver: str, message_type: str, content: Dict[str, Any]):
        """发送消息给其他智能体"""
        import time
        import uuid
        
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )
        self.communication_log.append(message)
        return message
    
    def receive_message(self, message: AgentMessage):
        """接收消息"""
        self.message_queue.append(message)
    
    def process_messages(self):
        """处理接收到的消息"""
        for message in self.message_queue:
            self._handle_message(message)
        self.message_queue.clear()
    
    def _handle_message(self, message: AgentMessage):
        """处理单个消息"""
        if message.message_type == "theory_hypothesis":
            self._handle_theory_hypothesis(message)
        elif message.message_type == "experiment_result":
            self._handle_experiment_result(message)
        elif message.message_type == "feedback":
            self._handle_feedback(message)
        elif message.message_type == "coordination":
            self._handle_coordination(message)
    
    def _handle_theory_hypothesis(self, message: AgentMessage):
        """处理理论假设消息"""
        pass
    
    def _handle_experiment_result(self, message: AgentMessage):
        """处理实验结果消息"""
        pass
    
    def _handle_feedback(self, message: AgentMessage):
        """处理反馈消息"""
        pass
    
    def _handle_coordination(self, message: AgentMessage):
        """处理协调消息"""
        pass
