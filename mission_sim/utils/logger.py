# mission_sim/utils/logger.py
"""
MCPC 框架高性能科学数据记录器 - 重构版
完全修复了控制力数组形状验证问题，增强数据健壮性
"""

import os
import h5py
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union


class HDF5Logger:
    """
    科学数据记录器 - 基于 HDF5 的高性能实现
    重构版：修复了 control_force 参数验证问题，增强数据类型容错
    
    特性：
    1. 内存缓冲 + 增量写入，避免频繁 I/O
    2. 支持数据压缩，减少存储空间
    3. 自动数据类型检测和优化
    4. 元数据记录，包含仿真配置信息
    5. 自动文件管理和版本控制
    """
    
    def __init__(self, 
                 filepath: str = "data/simulation.h5", 
                 buffer_size: int = 1000,
                 compression: bool = True,
                 auto_flush: bool = True):
        """
        初始化 HDF5 记录器
        
        Args:
            filepath: HDF5 文件保存路径
            buffer_size: 内存缓冲区大小（记录条数）
            compression: 是否启用数据压缩
            auto_flush: 是否在每次记录后自动检查并刷新缓冲区
        """
        self.filepath = os.path.abspath(filepath)
        self.buffer_size = buffer_size
        self.compression = compression
        self.auto_flush = auto_flush
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 如果文件已存在，先备份然后删除
        if os.path.exists(filepath):
            self._backup_existing_file()
            os.remove(filepath)
        
        # 初始化内存缓冲区
        self._init_buffers()
        
        # 记录器状态
        self.is_initialized = False
        self.total_steps = 0
        self.creation_time = datetime.now().isoformat()
        
        print(f"[HDF5Logger] 初始化完成，文件: {filepath}")
        print(f"[HDF5Logger] 缓冲区大小: {buffer_size}, 压缩: {compression}")
    
    def _init_buffers(self):
        """初始化内存缓冲区数据结构"""
        self.buffers = {
            'epochs': [],
            'nominal_states': [],
            'true_states': [],
            'nav_states': [],
            'tracking_errors': [],
            'control_forces': [],
            'accumulated_dvs': []
        }
        self.buffer_count = 0
    
    def _backup_existing_file(self):
        """备份已存在的文件"""
        backup_path = self.filepath + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            os.rename(self.filepath, backup_path)
            print(f"[HDF5Logger] 已备份旧文件: {backup_path}")
        except Exception as e:
            print(f"[HDF5Logger] 备份失败: {e}")
    
    def initialize_file(self, metadata: Optional[Dict[str, Any]] = None):
        """
        初始化 HDF5 文件结构
        
        Args:
            metadata: 仿真元数据字典
        """
        with h5py.File(self.filepath, 'w') as f:
            # 创建数据集（可扩展的第一维）
            for key in self.buffers.keys():
                # 根据数据类型确定形状
                if key == 'epochs':
                    shape = (0,)
                    dtype = np.float64
                elif key == 'accumulated_dvs':
                    shape = (0,)
                    dtype = np.float64
                else:
                    # 状态向量相关数据
                    shape = (0, 6) if 'state' in key or 'error' in key else (0, 3)
                    dtype = np.float64
                
                # 压缩设置
                compression_opts = 'gzip' if self.compression else None
                compression_lvl = 4 if self.compression else None
                
                f.create_dataset(
                    key,
                    shape=shape,
                    maxshape=(None,) + shape[1:],
                    dtype=dtype,
                    chunks=True,
                    compression=compression_opts,
                    compression_opts=compression_lvl
                )
            
            # 存储元数据
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        f.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        f.create_dataset(f"metadata/{key}", data=value)
                    else:
                        # 复杂对象序列化为字符串
                        f.attrs[key] = str(value)
            
            # 记录器元数据
            f.attrs['logger_creation_time'] = self.creation_time
            f.attrs['logger_buffer_size'] = self.buffer_size
            f.attrs['logger_compression'] = self.compression
        
        self.is_initialized = True
        print(f"[HDF5Logger] 文件结构初始化完成")
    
    def log_step(self, 
                 epoch: float,
                 nominal_state: np.ndarray,
                 true_state: np.ndarray,
                 nav_state: np.ndarray,
                 tracking_error: np.ndarray,
                 control_force: Union[np.ndarray, float],
                 accumulated_dv: float):
        """
        记录单步仿真数据 - 重构版
        修复了 control_force 参数验证问题，支持标量和数组
        
        Args:
            epoch: 当前仿真时间 (秒)
            nominal_state: 标称状态 [x,y,z,vx,vy,vz]
            true_state: 真实物理状态 [x,y,z,vx,vy,vz]
            nav_state: 导航滤波状态 [x,y,z,vx,vy,vz]
            tracking_error: 跟踪误差 [dx,dy,dz,dvx,dvy,dvz]
            control_force: 控制力向量 [Fx,Fy,Fz] 或标量
            accumulated_dv: 累计 ΔV 消耗
        """
        # 验证和标准化输入数据
        validated_data = self._validate_and_standardize_inputs(
            epoch, nominal_state, true_state, nav_state, 
            tracking_error, control_force, accumulated_dv
        )
        
        epoch, nominal_state, true_state, nav_state, tracking_error, control_force, accumulated_dv = validated_data
        
        # 存储到缓冲区
        self.buffers['epochs'].append(float(epoch))
        self.buffers['nominal_states'].append(nominal_state.astype(np.float64))
        self.buffers['true_states'].append(true_state.astype(np.float64))
        self.buffers['nav_states'].append(nav_state.astype(np.float64))
        self.buffers['tracking_errors'].append(tracking_error.astype(np.float64))
        self.buffers['control_forces'].append(control_force.astype(np.float64))
        self.buffers['accumulated_dvs'].append(float(accumulated_dv))
        
        self.buffer_count += 1
        self.total_steps += 1
        
        # 自动刷新检查
        if self.auto_flush and self.buffer_count >= self.buffer_size:
            self.flush()
    
    def _validate_and_standardize_inputs(self, epoch, *inputs):
        """
        验证和标准化输入数据 - 重构版
        修复了 control_force 参数验证问题
        
        Args:
            epoch: 时间
            inputs: 其他输入参数
            
        Returns:
            标准化后的参数元组
        """
        # 参数名称对应
        param_names = ['nominal_state', 'true_state', 'nav_state', 
                      'tracking_error', 'control_force', 'accumulated_dv']
        
        # 验证 epoch
        if not isinstance(epoch, (int, float)):
            raise TypeError(f"epoch 必须是数值类型，当前类型: {type(epoch)}")
        
        # 处理其他参数
        processed_inputs = []
        
        for i, (param_name, value) in enumerate(zip(param_names, inputs)):
            if param_name in ['nominal_state', 'true_state', 'nav_state', 'tracking_error']:
                # 状态向量：必须是形状为 (6,) 的数组
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"{param_name} 必须是 numpy 数组，当前类型: {type(value)}")
                
                if value.shape != (6,):
                    # 尝试重塑
                    if value.size == 6:
                        value = value.reshape(6)
                    else:
                        raise ValueError(f"{param_name} 必须是形状为 (6,) 的向量，当前形状: {value.shape}")
                
                processed_inputs.append(value)
            
            elif param_name == 'control_force':
                # 控制力：可以是标量或数组，自动转换为形状为 (3,) 的数组
                value = self._standardize_control_force(value)
                processed_inputs.append(value)
            
            elif param_name == 'accumulated_dv':
                # 累计 ΔV：必须是标量
                if not isinstance(value, (int, float, np.number)):
                    raise TypeError(f"accumulated_dv 必须是数值类型，当前类型: {type(value)}")
                processed_inputs.append(float(value))
        
        return (epoch,) + tuple(processed_inputs)
    
    def _standardize_control_force(self, control_force):
        """
        标准化控制力输入
        支持标量、1维数组、3维数组等多种格式
        
        Args:
            control_force: 原始控制力输入
            
        Returns:
            标准化后的控制力数组 (3,)
        """
        if isinstance(control_force, (int, float, np.number)):
            # 标量 -> 转换为数组 [force, 0, 0]
            return np.array([float(control_force), 0.0, 0.0], dtype=np.float64)
        
        elif isinstance(control_force, np.ndarray):
            if control_force.shape == ():
                # 0维数组 -> 转换为标量数组
                return np.array([float(control_force), 0.0, 0.0], dtype=np.float64)
            elif control_force.shape == (1,):
                # 1维标量数组 -> 转换为3维数组
                return np.array([float(control_force[0]), 0.0, 0.0], dtype=np.float64)
            elif control_force.shape == (3,):
                # 已经是3维数组
                return control_force.astype(np.float64)
            elif len(control_force) >= 3:
                # 长度≥3的数组 -> 取前3个元素
                return control_force[:3].astype(np.float64)
            else:
                # 其他形状 -> 尝试重塑为3维
                try:
                    if control_force.size == 3:
                        return control_force.reshape(3).astype(np.float64)
                    else:
                        # 无法处理，返回零向量
                        print(f"⚠️ 警告: 无法处理 control_force 形状 {control_force.shape}，使用零向量")
                        return np.zeros(3, dtype=np.float64)
                except:
                    print(f"⚠️ 警告: 处理 control_force 时出错，使用零向量")
                    return np.zeros(3, dtype=np.float64)
        
        else:
            # 未知类型，返回零向量
            print(f"⚠️ 警告: control_force 类型 {type(control_force)} 不支持，使用零向量")
            return np.zeros(3, dtype=np.float64)
    
    def flush(self):
        """
        将缓冲区数据写入磁盘
        返回: 实际写入的记录数
        """
        if self.buffer_count == 0:
            return 0
        
        # 确保文件已初始化
        if not self.is_initialized:
            self.initialize_file()
        
        try:
            with h5py.File(self.filepath, 'a') as f:
                for key, data_list in self.buffers.items():
                    if not data_list:  # 跳过空列表
                        continue
                    
                    data_array = np.array(data_list)
                    dataset = f[key]
                    
                    # 扩展数据集
                    current_size = dataset.shape[0]
                    new_size = current_size + len(data_array)
                    dataset.resize(new_size, axis=0)
                    
                    # 写入数据
                    dataset[current_size:new_size] = data_array
                    
                    # 更新属性
                    dataset.attrs['total_records'] = new_size
                    dataset.attrs['last_update'] = datetime.now().isoformat()
            
            written_count = self.buffer_count
            print(f"[HDF5Logger] 已写入 {written_count} 条记录，总计 {self.total_steps} 条")
            
            # 清空缓冲区
            self._init_buffers()
            
            return written_count
            
        except Exception as e:
            print(f"[HDF5Logger] 写入失败: {e}")
            # 保留缓冲区数据，下次尝试
            return 0
    
    def close(self):
        """
        关闭记录器，确保所有数据写入磁盘
        返回: 是否成功关闭
        """
        try:
            # 刷新剩余缓冲区数据
            if self.buffer_count > 0:
                self.flush()
            
            # 添加完成时间戳
            with h5py.File(self.filepath, 'a') as f:
                f.attrs['logger_close_time'] = datetime.now().isoformat()
                f.attrs['logger_total_steps'] = self.total_steps
                
                # 计算文件统计信息
                total_size = os.path.getsize(self.filepath)
                f.attrs['file_size_bytes'] = total_size
                f.attrs['file_size_mb'] = total_size / (1024 * 1024)
            
            print(f"[HDF5Logger] 记录器已关闭，文件: {self.filepath}")
            print(f"[HDF5Logger] 总计记录: {self.total_steps} 条")
            
            return True
            
        except Exception as e:
            print(f"[HDF5Logger] 关闭失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记录器统计信息
        
        返回: 包含统计信息的字典
        """
        if not os.path.exists(self.filepath):
            return {"status": "文件不存在"}
        
        stats = {
            "filepath": self.filepath,
            "total_steps": self.total_steps,
            "buffer_count": self.buffer_count,
            "file_exists": os.path.exists(self.filepath)
        }
        
        if os.path.exists(self.filepath):
            try:
                with h5py.File(self.filepath, 'r') as f:
                    stats["file_size_mb"] = os.path.getsize(self.filepath) / (1024 * 1024)
                    stats["creation_time"] = f.attrs.get('logger_creation_time', '未知')
                    
                    # 获取各数据集的记录数
                    for key in self.buffers.keys():
                        if key in f:
                            stats[f"{key}_records"] = f[key].shape[0]
            except Exception as e:
                stats["error"] = str(e)
        
        return stats
    
    def load_data(self, dataset_name: str, start_idx: int = 0, end_idx: int = None) -> np.ndarray:
        """
        从 HDF5 文件加载特定数据集
        
        Args:
            dataset_name: 数据集名称
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
            
        Returns: 加载的数据数组
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"HDF5 文件不存在: {self.filepath}")
        
        with h5py.File(self.filepath, 'r') as f:
            if dataset_name not in f:
                available = list(f.keys())
                raise KeyError(f"数据集 '{dataset_name}' 不存在。可用数据集: {available}")
            
            dataset = f[dataset_name]
            if end_idx is None:
                end_idx = dataset.shape[0]
            
            return dataset[start_idx:end_idx]
    
    def load_all_data(self) -> Dict[str, np.ndarray]:
        """
        加载所有数据集到内存
        
        Returns: 包含所有数据集的字典
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"HDF5 文件不存在: {self.filepath}")
        
        data = {}
        with h5py.File(self.filepath, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][:]
        
        return data
    
    def __del__(self):
        """析构函数，确保资源正确释放"""
        if hasattr(self, 'buffer_count') and self.buffer_count > 0:
            print(f"[HDF5Logger] 析构函数: 尝试刷新剩余 {self.buffer_count} 条记录")
            try:
                self.close()
            except:
                pass
    
    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (f"HDF5Logger(file={self.filepath}, "
                f"total_steps={self.total_steps}, "
                f"buffer={self.buffer_count}/{self.buffer_size})")


class SimulationMetadata:
    """
    仿真元数据管理器
    用于记录和存储仿真的配置和参数
    """
    
    @staticmethod
    def create_mission_metadata(mission_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建任务元数据
        
        Args:
            mission_name: 任务名称
            config: 仿真配置字典
            
        Returns: 标准化元数据字典
        """
        metadata = {
            "mission_name": mission_name,
            "simulation_timestamp": datetime.now().isoformat(),
            "software_version": "MCPC-Framework v1.0",
            "simulation_config": config,
            "hardware_info": {
                "platform": os.name,
                "processor": os.cpu_count(),
                "python_version": os.sys.version
            }
        }
        return metadata
