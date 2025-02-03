from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, safe_unicode_decode
import os
import torch
import hepai as hai
import numpy as np
import asyncio
from FlagEmbedding import BGEM3FlagModel

async def a_llm_model_func(
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

async def a_embedding_func(
        texts: list[str]
        ) -> np.ndarray:
    # 设置使用的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    # 加载模型到指定GPU
    embedding_model_path = "/data/szj/hai-rag-OS/embedding/BAAI/bge-m3/"
    model = BGEM3FlagModel(embedding_model_path,
                          use_fp16=True,  # 开启FP16加速推理
                          device=device)  # 指定使用的GPU设备
    # 异步执行推理
    embeddings = await asyncio.to_thread(
        model.encode, 
        sentences = texts, 
        batch_size=16, 
        max_length=8192
        )
    return embeddings['dense_vecs']


# def embedding_func(texts: list[str]) -> np.ndarray:
#     # 设置使用的GPU设备
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     device = torch.device("cuda:0")
#     # 加载模型到指定GPU
#     embedding_model_path = "/data/szj/hai-rag-OS/embedding/BAAI/bge-m3/"
#     model = BGEM3FlagModel(embedding_model_path,
#                           use_fp16=True,  # 开启FP16加速推理
#                           device=device)  # 指定使用的GPU设备
#     # 同步执行推理
#     embeddings = model.encode(
#         sentences=texts,
#         batch_size=16,
#         max_length=8192)
#     return embeddings['dense_vecs']

# def embedding_func(texts: list[str]) -> np.ndarray:
#     # 自动选择最佳GPU
#     try:
#         device_id = find_available_gpu(min_memory=2048)  # 至少需要2GB显存
#         device = torch.device(f"cuda:{device_id}")
#     except RuntimeError as e:
#         print(f"GPU selection failed: {e}, falling back to CPU")
#         device = torch.device("cpu")
#     # 加载模型到指定GPU
#     embedding_model_path = "/data/szj/hai-rag-OS/embedding/BAAI/bge-m3/"
#     model = BGEM3FlagModel(embedding_model_path,
#                           use_fp16=True,  # 开启FP16加速推理
#                           device=device)  # 指定使用的GPU设备
#     # 同步执行推理
#     embeddings = model.encode(
#         sentences=texts,
#         batch_size=16,
#         max_length=8192)
#     return embeddings['dense_vecs']


if __name__ == "__main__":

    # embedding test
    # texts = [
    #     "This is a test sentence.",
    #     "This is another test sentence.",
    #     "This is a test sentence for light rag."
    # ]
    # embeddings = embedding_func(texts)
    # print(embeddings.shape)
    # embeddings = asyncio.run(a_embedding_func(texts))
    # print(embeddings.shape)

    WORKING_DIR = "/home/xiongdb/Test/lightrag/lightrag_store"
    # api_key=os.getenv("HEPAI_API_KEY")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=a_llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=a_embedding_func
        )
    )

    with open("./files/README.md") as f:
        rag.insert(f.read())

    # # Perform naive search
    # print(rag.query("What are the top themes in the doc?", param=QueryParam(mode="naive")))

    # # Perform local search
    # print(rag.query("What are the top themes in the doc?", param=QueryParam(mode="local")))

    # # Perform global search
    # print(rag.query("What are the top themes in the doc?", param=QueryParam(mode="global")))

    # # Perform hybrid search
    # print(rag.query("What are the top themes in the doc?", param=QueryParam(mode="hybrid")))

    # Perform mix search (Knowledge Graph + Vector Retrieval)
    # Mix mode combines knowledge graph and vector search:
    # - Uses both structured (KG) and unstructured (vector) information
    # - Provides comprehensive answers by analyzing relationships and context
    # - Supports image content through HTML img tags
    # - Allows control over retrieval depth via top_k parameter
    print(rag.query("What are the LightRAG?", param=QueryParam(mode="mix")))
    pass