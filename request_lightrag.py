from hepai import HRModel

model = HRModel.connect(
    name="hepai/lightrag",
    base_url="http://localhost:4260/apiv2"
)

funcs = model.functions()  # Get all remote callable functions.
print(f"Remote callable funcs: {funcs}")

# 请求远程模型的custom_method方法
with open("./files/README.md", "r") as f:
    text = f.read()
    output = model.interface(
        query = "What is the LightRAG?",
        mode = "global",
        embedding_texts = text
    )

    print(output)