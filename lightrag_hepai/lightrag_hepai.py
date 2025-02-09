from typing import Dict, Union, Literal, List
import hepai as hai

# from lightrag.llm.openai import openai_complete_if_cache, openai_embed
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

import logging
logger = logging.getLogger(__name__)


# 加载本地.env文件
from dotenv import load_dotenv
load_dotenv()

# 寻找当前目录是否.env文件存在
error_attention = '''env file not found, please create one in lightrag_hepai directory with following content:
```env
# .env file
Embedding_model_path="/path/to/embedding_model"
embedding_dimension=your_embedding_dimension
max_token_size=your_max_token_size
gpu_id=your_gpu_id
llm_api_key="your_api_key"
llm_base_url="your_base_url"
llm_model_name="openai/gpt-4o-mini"
working_dir="../lightrag_store/"
```
'''
# 获取当前工作目录和脚本所在目录
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
env_in_script = os.path.exists(os.path.join(script_dir, ".env"))
absolute_path = os.path.join(script_dir, ".env")
print(f"The current environment path: {absolute_path}")
if not os.path.exists(absolute_path):
    raise FileNotFoundError(error_attention)


## 变量加载
Embedding_model_path = os.getenv("Embedding_model_path") # 嵌入模型路径
embedding_dimension = int(os.getenv("embedding_dimension"))  # 嵌入维度
max_token_size = int(os.getenv("max_token_size"))  # 最大token长度
gpu_id = int(os.getenv("gpu_id"))  # GPU设备ID
llm_api_key = os.getenv("llm_api_key")  # hepai api key
llm_base_url = os.getenv("llm_base_url")  # hepai base url
llm_model_name = os.getenv("llm_model_name")  # hepai model name
working_dir_env = os.getenv("working_dir")  # 缓存路径
if not os.path.exists(working_dir_env):
    os.makedirs(working_dir_env, exist_ok=True)

class LightRAG_HepAI:
    def __init__(self, working_dir = None):
        # 设置缓存路径
        self.working_dir = working_dir or working_dir_env

    # Define the openai_complete function for the LightRAG model.
    async def a_llm_model_func(
            self, 
            prompt, 
            model = llm_model_name,
            system_prompt=None, 
            history_messages=[], 
            api_key=llm_api_key, 
            base_url=llm_base_url,
            keyword_extraction=False, 
            **kwargs) -> str:
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # 调用hepai接口
        params = {
            "model": model,
            "messages": messages,
        }
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        params.update(kwargs)

        # 同步调用hepai接口
        hepai_client = hai.HepAI(api_key=api_key, base_url=base_url)
        response = hepai_client.chat.completions.create(**params)

        # 异步调用hepai接口
        # hepai_async_client = hai.AsyncHepAI(api_key=api_key, base_url=base_url)
        # if "response_format" in kwargs:
        #     response = await hepai_async_client.beta.chat.completions.parse(
        #         model=model, messages=messages, **kwargs
        #     )
        # else:
        #     response = await hepai_async_client.chat.completions.create(
        #         model=model, messages=messages, **kwargs
        # )

        logger.info(f"Call hepai model: {model}")
        # 异步迭代返回结果
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
            # 同步返回结果
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
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device(f"cuda:{gpu_id}")
        # 加载模型到指定GPU
        model = BGEM3FlagModel(
            model_name_or_path = Embedding_model_path,
            use_fp16=True,  # 开启FP16加速推理
            device=device)  # 指定使用的GPU设备
        # 同步执行推理
        embeddings = model.encode(
            sentences=texts,
            batch_size=16,
            max_length=max_token_size)
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

    def interface(
        self, 
        query: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
        embedding_texts: Union[str, List[str]] = None,
        custom_prompt: str = "",
        conversation_history: List[Dict] = [],
        history_turns: int = 3,
        embedding_dimension = embedding_dimension,
        user_name: str = "admin",
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
        
        # TODO: 增加查询超时的反馈和处理

        # 设置缓存路径
        WORKING_DIR = os.path.join(self.working_dir, user_name) # "../lightrag_store/"
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
                embedding_dim=embedding_dimension,
                max_token_size=max_token_size,
                func=self.a_embedding_func
                )
            )
        
        # 插入文档
        if embedding_texts:
            rag.insert(string_or_strings = embedding_texts)

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
    # 测试接口
    lightrag_hepai = LightRAG_HepAI()
    # with open("/home/xiongdb/test/MinerU/files/eece1cae-a7c0-4926-83d0-5c1877ef8651/2501/output/2501.md", "r") as f:
    with open("/home/xiongdb/VSproject/LightRAG/README.md", "r") as f:
        text = f.read()
        output = lightrag_hepai.interface(
            query = "What is the LightRAG?",
            # query = "How DeepSeek-R1 works?",
            mode = "hybrid",
            # embedding_texts = text
        )

        print(output)