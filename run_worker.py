from typing import Dict, Union, Literal, List
from dataclasses import dataclass, field
import json
import hepai as hai
from hepai import HRModel, HModelConfig, HWorkerConfig, HWorkerAPP

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, safe_unicode_decode
import os
import torch
import numpy as np
import asyncio
import nest_asyncio
nest_asyncio.apply()
lock = asyncio.Lock()

from FlagEmbedding import BGEM3FlagModel

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

    permissions: str = field(default='users: admin', metadata={"help": "Model's permissions, separated by ;, e.g., 'groups: default; users: a, b; owner: c'"})
    description: str = field(default='This is a demo worker of HEP AI framework (HepAI)', metadata={"help": "Model's description"})
    author: str = field(default=None, metadata={"help": "Model's author"})
    debug: bool = field(default=True, metadata={"help": "Debug mode"})
    daemon: bool = field(default=False, metadata={"help": "Run as daemon"})

class CustomWorkerModel(HRModel):  # Define a custom worker model inheriting from HRModel.
    def __init__(self, config: HModelConfig):
        super().__init__(config=config)

    # Define the openai_complete function for the LightRAG model.
    async def a_llm_model_func(
            self, 
            prompt, 
            model = "openai/gpt-4o-mini",
            system_prompt=None, 
            history_messages=[], 
            api_key=os.getenv("HEPAI_API_KEY2"), 
            base_url=os.getenv("BASE_URL_V2"),
            keyword_extraction=False, 
            **kwargs) -> str:
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        hepai_client = hai.HepAI(api_key=api_key, base_url=base_url)

        # 调用hepai接口
        params = {
            "model": model,
            "messages": messages,
        }
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        params.update(kwargs)
        response = hepai_client.chat.completions.create(**params)

        if hasattr(response, "__aiter__"):
            async def inner():
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))
                    yield content

            return inner()
        else:
            content = response.choices[0].message.content
            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))
            return content
        
        # 调用openai接口
        # return await openai_complete_if_cache(
        #     prompt = prompt,
        #     model = model,
        #     system_prompt=system_prompt,
        #     history_messages=history_messages,
        #     api_key=api_key,
        #     base_url=base_url,
        #     **kwargs
        # )
    
    # Define the embeddings function for the LightRAG model.
    async def a_embedding_func(
            self, 
            texts: list[str]) -> np.ndarray:
        # 设置使用的GPU设备
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
        # 加载模型到指定GPU
        embedding_model_path = "/data/szj/hai-rag-OS/embedding/BAAI/bge-m3/"
        model = BGEM3FlagModel(
            embedding_model_path,
            use_fp16=True,  # 开启FP16加速推理
            device=device)  # 指定使用的GPU设备
        # 同步执行推理
        embeddings = model.encode(
            sentences=texts,
            batch_size=16,
            max_length=8192)
        # 异步执行推理
        # embeddings = await asyncio.to_thread(
        #     model.encode, 
        #     sentences = texts, 
        #     batch_size=16, 
        #     max_length=8192
        #     )
        return embeddings['dense_vecs']
    
    # 插入文档
    async def insert_documents(
            self, 
            rag: LightRAG, 
            embedding_texts: 
            Union[str, List[str]]):
        await rag.ainsert(string_or_strings=embedding_texts)
    
    # 查询文档
    async def query_documents(
            self, 
            rag: LightRAG, 
            query: str,
            custom_prompt: str,
            query_param: QueryParam) -> str:
        response = await rag.aquery(
            query=query, 
            prompt=custom_prompt, 
            param=query_param)
        return response


    @HRModel.remote_callable  # Decorate the function to enable remote call.
    def interface(
        self, 
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global",
        embedding_texts: Union[str, List[str]] = None,
        custom_prompt: str = "",
        conversation_history: List[Dict] = [],
        history_turns: int = 3,
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
        Returns:
            The generated response string.
        """
        
        # 设置缓存路径
        WORKING_DIR = "./lightrag_store/"
        ## 获取绝对路径
        WORKING_DIR = os.path.abspath(WORKING_DIR)
        ## 判断文件夹是否存在
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)

        # 设置LightRAG instance
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=self.a_llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=self.a_embedding_func
                )
            )
        
        # 插入文档
        rag.insert(string_or_strings = embedding_texts)
        # self.insert_documents(rag, embedding_texts)

        # 设置查询参数
        query_param = QueryParam(
            mode=mode,
            conversation_history=conversation_history,
            history_turns = history_turns
        )
        # response = self.query_documents(rag, query, custom_prompt, query_param)
        response = rag.query(query=query, prompt=custom_prompt, param=query_param)
        try:
            return response
        except Exception as e:
            return f"LightRAG model call failed with Error: {e}"


if __name__ == "__main__":

    import uvicorn
    from fastapi import FastAPI
    model_config, worker_config = hai.parse_args((CustomModelConfig, CustomWorkerConfig))
    model = CustomWorkerModel(model_config)  # Instantiate the custom worker model.
    app: FastAPI = HWorkerAPP(model, worker_config=worker_config)  # Instantiate the APP, which is a FastAPI application.

    print(app.worker.get_worker_info(), flush=True)
    # 启动服务
    uvicorn.run(app, host=app.host, port=app.port)