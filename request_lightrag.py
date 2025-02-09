from hepai import HRModel

model = HRModel.connect(
    name="hepai/lightrag",
    base_url="http://localhost:4260/apiv2"
    # api_key=os.environ.get('HEPAI_API_KEY2'),
    # base_url="https://aiapi001.ihep.ac.cn/apiv2"
)

funcs = model.functions() # Get all remote callable functions.
print(f"Remote callable funcs: {funcs}")

# 请求远程模型的custom_method方法
# with open("/home/xiongdb/VSproject/LightRAG/README.md", "r") as f:
with open("/home/xiongdb/test/MinerU/files/eece1cae-a7c0-4926-83d0-5c1877ef8651/2501/output/2501.md", "r") as f:
    text = f.read()
    output = model.interface(
        # query = "What is the LightRAG?",
        query = "what is the DeepSeek-R1?",
        mode = "mix",
        embedding_texts = text,
        # task_id ="2e473e75-713e-4dda-94b2-d62f04c1d16f"
    )

    print(output)
