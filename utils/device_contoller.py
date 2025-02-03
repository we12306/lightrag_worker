import torch

def find_available_gpu(min_memory: int = 1024) -> int:
    """自动寻找可用GPU的逻辑实现"""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-compatible GPU found")

    # 获取GPU数量
    device_count = torch.cuda.device_count()
    
    # 评估各GPU内存情况
    best_gpu = None
    max_memory = 0
    
    for device_id in range(device_count):
        try:
            # 获取显存信息（单位：MB）
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            free_mem_mb = free_mem // 1024**2
            
            # 只考虑满足最小内存要求的设备
            if free_mem_mb >= min_memory and free_mem_mb > max_memory:
                max_memory = free_mem_mb
                best_gpu = device_id
        except RuntimeError:
            continue  # 跳过不可访问的GPU

    if best_gpu is not None:
        print(f"Selected GPU {best_gpu} with {max_memory}MB free memory")
        return best_gpu
    else:
        raise RuntimeError(f"No available GPU with at least {min_memory}MB memory")