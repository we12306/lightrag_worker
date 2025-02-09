import os
from pathlib import Path

# 获取当前项目的根目录
## 家目录
home_path = str(Path.home())
dir_path = os.path.join(home_path, ".Dr.Sai/lightrag_hepai")
task_store_dir: str = os.path.join(dir_path, "task_manage")

import json
import uuid
import fcntl
from datetime import datetime
from typing import List, Dict, Optional, Union, Literal
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


class AtomicFileWriter:
    """原子文件操作类"""
    @staticmethod
    def safe_write(file_path: str, data: dict):
        """带文件锁的原子写入"""
        with open(file_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁
            try:
                json.dump(data, f, indent=2)
                f.flush()  # 强制写入磁盘
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def safe_append(file_path: str, content: str):
        """带文件锁的追加写入"""
        with open(file_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(content)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

class TaskManager:
    """增强版任务管理系统"""
    def __init__(
            self, 
            manage_dir: str = task_store_dir):
        
        self.manage_dir = manage_dir
        self.executor = ThreadPoolExecutor(max_workers=4)  # 线程池管理任务
        os.makedirs(manage_dir, exist_ok=True)

    def _get_task_path(self, task_id: str) -> str:
        return os.path.join(self.manage_dir, f"{task_id}")

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def create_task(self, task_id: str, query: str, **params) -> Dict:
        """创建新任务并初始化文件"""
        task_path = self._get_task_path(task_id)
        os.makedirs(task_path, exist_ok=True)
        
        task_info = {
            "task_id": task_id,
            "query": query,
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "start_time": datetime.now().isoformat(),
            "params": params,
            "log_file": os.path.join(task_path, "task.log"),
            "data_file": os.path.join(task_path, "task.json")
        }
        
        # 原子化写入操作
        AtomicFileWriter.safe_append(task_info["log_file"], f"[{datetime.now().isoformat()}] Task created\n")
        AtomicFileWriter.safe_write(task_info["data_file"], task_info)
        
        return task_info

    def update_task(self, task_id: str, **update_fields) -> bool:
        """原子化更新任务状态"""
        data_file = os.path.join(self._get_task_path(task_id), "task.json")
        if not os.path.exists(data_file):
            return False

        # 读取-修改-写入原子操作
        with open(data_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            task_info = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)

        task_info.update(update_fields)
        if "status" in update_fields and update_fields["status"] in ["completed", "failed"]:
            task_info["end_time"] = datetime.now().isoformat()
        
        AtomicFileWriter.safe_write(data_file, task_info)
        return True

    def append_log(self, task_id: str, message: str):
        """线程安全的日志追加"""
        log_file = os.path.join(self._get_task_path(task_id), "task.log")
        AtomicFileWriter.safe_append(log_file, f"[{datetime.now().isoformat()}] {message}\n")

    @lru_cache(maxsize=1000)
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """带缓存的状态查询"""
        data_file = os.path.join(self._get_task_path(task_id), "task.json")
        if not os.path.exists(data_file):
            return None
        
        with open(data_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)  # 共享锁
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)