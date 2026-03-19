# mission_sim/utils/loggers.py
import h5py
import numpy as np
import os

class HDF5Logger:
    """高性能增量式 HDF5 数据记录器"""
    def __init__(self, filepath, flush_interval=500):
        self.filepath = filepath
        self.flush_interval = flush_interval
        self._buffer = {}
        self._step_count = 0
        
        # 每次仿真开始前，强制清理旧文件，防止数据污染
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def set_metadata(self, group_name, meta_dict):
        """记录仿真环境的静态配置"""
        with h5py.File(self.filepath, 'a') as f:
            grp = f.require_group(group_name)
            for k, v in meta_dict.items():
                grp[k] = v

    def log(self, group: str, dataset: str, data: np.ndarray):
        """将当前步的数据存入内存 Buffer"""
        key = f"{group}/{dataset}"
        if key not in self._buffer:
            self._buffer[key] = []
        # 必须使用 np.copy 防止引用同一块内存地址
        self._buffer[key].append(np.copy(data))

    def step(self):
        """主循环推进，达到阈值后自动落盘 (防 OOM 核心)"""
        self._step_count += 1
        if self._step_count % self.flush_interval == 0:
            self.flush()

    def flush(self):
        """将 Buffer 中的数据增量追加到硬盘"""
        if not self._buffer:
            return
            
        with h5py.File(self.filepath, 'a') as f:
            for key, data_list in self._buffer.items():
                group_name, dataset_name = key.split('/')
                grp = f.require_group(group_name)
                
                new_data = np.array(data_list)
                
                if dataset_name not in grp:
                    # 首次创建支持无限追加的数据集
                    maxshape = (None,) + new_data.shape[1:]
                    grp.create_dataset(dataset_name, data=new_data, maxshape=maxshape, chunks=True)
                else:
                    # 扩容并追加数据
                    dset = grp[dataset_name]
                    old_len = dset.shape[0]
                    dset.resize(old_len + new_data.shape[0], axis=0)
                    dset[old_len:] = new_data
                    
        # 清空内存缓冲
        self._buffer.clear()

    def close(self):
        """仿真结束时强制写出剩余数据"""
        self.flush()

