
import hepai as hai
from hepai import  HModelConfig, HWorkerConfig, HWorkerAPP, HRModel
from lightrag_hepai.lightrag_hepai import LightRAG_HepAI
from lightrag_hepai.utils.task_manager import TaskManager

from typing import Dict, Union, Literal, List, Optional
from dataclasses import dataclass, field
import threading


class LightRAGWorkerModel(HRModel):  # Define a custom worker model inheriting from HRModel.
    def __init__(self, config: HModelConfig):
        super().__init__(config=config)
        
        self.task_manager = TaskManager()
        self.shutdown_flag = threading.Event()  # 安全关闭标志
    
    def _background_processing(self, task_id: str, query: str, **kwargs):
        """后台处理任务的强化版本"""
        try:
            self.task_manager.update_task(task_id, status="running", progress=10)
            self.task_manager.append_log(task_id, "Starting processing")
            
            # 实际处理逻辑
            lightrag_hepai = LightRAG_HepAI()
            result = lightrag_hepai.interface(query=query, **kwargs)
            
            # 定期检查关闭标志
            if self.shutdown_flag.is_set():
                self.task_manager.append_log(task_id, "Processing interrupted by shutdown")
                return

            self.task_manager.update_task(
                task_id,
                status="completed",
                progress=100,
                result=result
            )
            self.task_manager.append_log(task_id, "Processing completed successfully")
            
        except Exception as e:
            self.task_manager.update_task(
                task_id,
                status="failed",
                error=str(e),
                progress=100
            )
            self.task_manager.append_log(task_id, f"Processing failed: {str(e)}")

    async def graceful_shutdown(self):
        """安全关闭方法"""
        self.shutdown_flag.set()
        self.task_manager.executor.shutdown(wait=True)

    @HRModel.remote_callable  # Decorate the function to enable remote call.
    def interface(
        self, 
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global",
        embedding_texts: Union[str, List[str]] = None,
        custom_prompt: str = "",
        conversation_history: List[Dict] = [],
        history_turns: int = 3,
        user_name: str = "admin",
        timeout: int = 60,
        task_id: Optional[str] = None,
        **kwargs) -> str:
        """
        Call the LightRAG model to generate a response.
        Args:
            query: The input query.
            mode: The LightRAG mode, can be "local", "global", "hybrid", "naive", "mix".
            embedding_texts: The texts used to generate the embeddings.
            custom_prompt: The custom prompt for the lightRAG model.
            conversation_history: The conversation history.
            history_turns: The number of conversation turns to use for the conversation history.
            user_name: The user name.
            timeout: The query timeout.
            task_id: The task id.
            **kwargs: Other keyword arguments.
        Returns:
            The generated response string.
        """
        
        # 检查是否存在相同任务
        if task_id:
            existing_task: dict = self.task_manager.get_task_status(task_id)
            if existing_task:
                return self._format_response(existing_task)
        else:
            task_id = self.task_manager.generate_task_id()

        # 初始化任务
        task_info = self.task_manager.create_task(
            task_id=task_id,
            query=query,
            mode=mode,
            embedding_texts = embedding_texts[:100]+"..." if isinstance(embedding_texts, str) else embedding_texts[0][:100]+"..." if isinstance(embedding_texts, list) else None,
            custom_prompt=custom_prompt,
            conversation_history=conversation_history,
            history_turns=history_turns,
            user_name=user_name,
            timeout=timeout,
            **kwargs
        )

        # 使用线程池提交任务
        future = self.task_manager.executor.submit(
            self._background_processing,
            task_id=task_id,
            query=query,
            mode=mode,
            embedding_texts = embedding_texts,
            custom_prompt=custom_prompt,
            conversation_history=conversation_history,
            history_turns=history_turns,
            user_name=user_name,
            **kwargs
        )
        
        try:
            # 等待结果或超时
            result = future.result(timeout=timeout)
            # 检查处理结果
            current_status = self.task_manager.get_task_status(task_id)
            if current_status["status"] == "completed":
                return current_status["result"]
        # except TimeoutError:
        #     # 超时处理
        #     current_status = self.task_manager.get_task_status(task_id)
        #     return self._handle_timeout(task_id, current_status)
        except Exception as e:
            if not str(e):
                # 超时处理
                current_status = self.task_manager.get_task_status(task_id)
                return self._handle_timeout(task_id, current_status)
            else:
                return f"Task failed: {str(e)}"

        # # 启动后台线程
        # thread = threading.Thread(
        #     target=self._background_task,
        #     args=(task_id, query),
        #     kwargs={
        #         "mode": mode,
        #         "embedding_texts": embedding_texts,
        #         "custom_prompt": custom_prompt,
        #         "conversation_history": conversation_history,
        #         "history_turns": history_turns,
        #         "user_name": user_name,
        #         **kwargs
        #     }
        # )
        # self.running_tasks[task_id] = thread
        # thread.start()

        # # 等待结果或超时
        # thread.join(timeout=timeout)

        # # 检查处理结果
        # current_status = self.task_manager.get_task_status(task_id)
        # if current_status["status"] == "completed":
        #     return current_status["result"]
        
        # # 超时处理
        # return self._handle_timeout(task_id, current_status)

    def _format_response(self, task_info: Dict) -> str:
        """格式化任务状态响应"""
        status = task_info["status"]
        if status == "completed":
            return task_info["result"]
        
        log_file = task_info["log_file"]
        with open(log_file, "r") as f:
            logs = f.read()
        
        response = [
            f"Task {task_info['task_id']} status: {status}",
            f"Progress: {task_info.get('progress', 0)}%",
            f"Logs:\n{logs[-2000:]}"  # 返回最后2000字符的日志
        ]
        
        if task_info.get("error"):
            response.append(f"Error: {task_info['error']}")
        
        return "\n\n".join(response)

    def _handle_timeout(self, task_id: str, current_status: Dict) -> str:
        """处理超时情况"""
        base_response = [
            "Request is processing in background",
            "You can check status later using this ID in the following format:",
            f"```Task ID: <{task_id}>```"
        ]
        
        if current_status["status"] == "failed":
            return f"Task failed immediately: {current_status.get('error', 'Unknown error')}"
        
        # 获取部分处理结果（如果有）
        partial_result = current_status.get("result")
        if partial_result:
            base_response.insert(0, "Partial result:")
            base_response.insert(1, partial_result)
        
        return "\n".join(base_response)



@dataclass
class CustomModelConfig(HModelConfig):
    name: str = field(default="hepai/lightrag", metadata={"help": "Model's name"})
    permission: Union[str, Dict] = field(default=None, metadata={"help": "Model's permission, separated by ;, e.g., 'groups: all; users: a, b; owner: c', will inherit from worker permissions if not setted"})
    version: str = field(default="2.0", metadata={"help": "Model's version"})

@dataclass
class CustomWorkerConfig(HWorkerConfig):
    host: str = field(default="0.0.0.0", metadata={"help": "Worker's address, enable to access from outside if set to `0.0.0.0`, otherwise only localhost can access"})
    port: int = field(default=4260, metadata={"help": "Worker's port, default is None, which means auto start from `auto_start_port`"})
    auto_start_port: int = field(default=42602, metadata={"help": "Worker's start port, only used when port is set to `auto`"})
    route_prefix: str = field(default="/apiv2", metadata={"help": "Route prefix for worker"})
    controller_address: str = field(default="http://localhost:42601", metadata={"help": "Controller's address"})
    # controller_address: str = field(default="https://aiapi001.ihep.ac.cn", metadata={"help": "Controller's address"})
    controller_prefix: str = field(default="/apiv2", metadata={"help": "Controller's route prefix"})
    
    speed: int = field(default=1, metadata={"help": "Model's speed"})
    limit_model_concurrency: int = field(default=5, metadata={"help": "Limit the model's concurrency"})
    stream_interval: float = field(default=0., metadata={"help": "Extra interval for stream response"})
    no_register: bool = field(default=True, metadata={"help": "Do not register to controller"})
    permissions: str = field(default='users: admin; owner: xiongdb@ihep.ac.cn', metadata={"help": "Model's permissions, separated by ;, e.g., 'groups: default; users: a, b; owner: c'"})
    description: str = field(default='This is a demo worker of HEP AI framework (HepAI)', metadata={"help": "Model's description"})
    author: str = field(default="xiongdb@ihep.ac.cn", metadata={"help": "Model's author"})
    api_key: str = field(default="", metadata={"help": "API key for reigster to controller, ensure the security"})
    debug: bool = field(default=True, metadata={"help": "Debug mode"})
    type: Literal["llm", "actuator", "preceptor", "memory", "common"] = field(default="common", metadata={"help": "Specify worker type, could be help in some cases"})
    daemon: bool = field(default=False, metadata={"help": "Run as daemon"})


if __name__ == "__main__":

    import uvicorn
    from fastapi import FastAPI
    model_config, worker_config = hai.parse_args((CustomModelConfig, CustomWorkerConfig))
    model = LightRAGWorkerModel(model_config)  # Instantiate the custom worker model.
    app: FastAPI = HWorkerAPP(model, worker_config=worker_config)  # Instantiate the APP, which is a FastAPI application.

    print(app.worker.get_worker_info(), flush=True)
    # 启动服务
    uvicorn.run(app, host=app.host, port=app.port)