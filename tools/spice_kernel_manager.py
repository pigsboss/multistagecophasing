"""
SPICE核文件管理器

自动化下载、更新和管理SPICE核文件（用于高精度星历计算）。
支持DE440行星历表、姿态文件、计时文件等。

特性：
1. 自动下载缺失的核文件
2. 版本管理和更新检查
3. 本地缓存，避免重复下载
4. 多线程下载支持
5. 验证文件完整性（MD5校验）

作者: MCPC开发团队
版本: 1.0.0
"""

import os
import sys
import time
import hashlib
import shutil
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from tqdm import tqdm
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("警告: 缺少requests或tqdm库，无法使用自动下载功能")
    print("请安装: pip install requests tqdm")

# SPICE核文件服务器URL
SPICE_SERVERS = {
    'naif': 'https://naif.jpl.nasa.gov/pub/naif/',
    'ssd': 'https://ssd.jpl.nasa.gov/ftp/eph/planets/',
    'pds': 'https://naif.jpl.nasa.gov/pub/naif/pds/'
}


@dataclass
class KernelInfo:
    """SPICE核文件信息"""
    name: str                    # 文件名
    category: str               # 类别：'ephemeris', 'attitude', 'frame', 'clock', 'generic'
    url: str                    # 下载URL
    local_path: Path           # 本地路径
    version: str = ""          # 版本号
    size: int = 0              # 文件大小（字节）
    md5: Optional[str] = None  # MD5校验和
    description: str = ""      # 文件描述
    required: bool = True      # 是否必需
    last_update: Optional[datetime] = None  # 最后更新时间
    compressed: bool = False   # 是否为压缩文件


@dataclass
class KernelConfig:
    """SPICE核配置"""
    kernel_dir: Path = Path("~/.mission_sim/spice_kernels").expanduser()  # 默认存储目录
    auto_download: bool = True     # 是否自动下载
    check_updates: bool = True     # 检查更新
    update_interval_days: int = 30  # 更新检查间隔
    verify_integrity: bool = True  # 验证文件完整性
    max_retries: int = 3           # 最大重试次数
    timeout: int = 30              # 下载超时（秒）
    parallel_downloads: int = 3    # 并行下载数
    verbose: bool = True           # 详细输出
    use_cache: bool = True         # 使用缓存
    cache_size_mb: int = 1024      # 缓存大小(MB)


class SPICEKernelManager:
    """
    SPICE核文件管理器
    
    自动化管理SPICE核文件的下载、更新和验证。
    """
    
    # 常用核文件定义
    COMMON_KERNELS = {
        'de440': {
            'name': 'de440.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp',
            'description': 'DE440行星历表（1900-2050）',
            'required': True,
            'size': 135_000_000,  # 约135MB
            'compressed': False
        },
        'de440_small': {
            'name': 'de440s.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp',
            'description': 'DE440简化版行星历表（1900-2050）',
            'required': False,
            'size': 7_000_000,  # 约7MB
            'compressed': False
        },
        'pck00010': {
            'name': 'pck00010.tpc',
            'category': 'frame',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc',
            'description': '行星常数和方向信息',
            'required': True,
            'size': 2_000_000,  # 约2MB
            'compressed': False
        },
        'naif0012': {
            'name': 'naif0012.tls',
            'category': 'clock',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls',
            'description': '时间系统转换',
            'required': True,
            'size': 100_000,  # 约100KB
            'compressed': False
        },
        'earth_200101': {
            'name': 'earth_200101_230317_230317.bpc',
            'category': 'attitude',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_230317_230317.bpc',
            'description': '地球姿态模型（2001-2023）',
            'required': False,
            'size': 5_000_000,  # 约5MB
            'compressed': False
        },
        'earth_070425': {
            'name': 'earth_070425_370426_predict.bpc',
            'category': 'attitude',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_070425_370426_predict.bpc',
            'description': '地球姿态预测模型',
            'required': False,
            'size': 3_000_000,  # 约3MB
            'compressed': False
        },
        'moon_pa': {
            'name': 'moon_pa_de440_200625.bpc',
            'category': 'attitude',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc',
            'description': '月球姿态模型',
            'required': False,
            'size': 2_000_000,  # 约2MB
            'compressed': False
        },
        'latest_leapseconds': {
            'name': 'latest_leapseconds.tls',
            'category': 'clock',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls',
            'description': '最新闰秒表',
            'required': False,
            'size': 50_000,  # 约50KB
            'compressed': False
        }
    }
    
    def __init__(self, config: Optional[KernelConfig] = None):
        """
        初始化SPICE核管理器
        
        Args:
            config: 核配置，如为None则使用默认配置
        """
        self.config = config or KernelConfig()
        self.kernel_dir = self.config.kernel_dir
        
        # 确保目录存在
        self.kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载核数据库
        self._kernels: Dict[str, KernelInfo] = {}
        self._load_kernel_database()
        
        # 状态文件路径
        self.state_file = self.kernel_dir / "kernel_state.json"
        self._load_state()
        
        # 锁，用于线程安全
        self._lock = threading.RLock()
        
        self.verbose = self.config.verbose
        
        if not HAS_DEPENDENCIES:
            if self.config.auto_download:
                warnings.warn("缺少requests或tqdm库，自动下载功能不可用")
        
        if self.verbose:
            print(f"[SPICEKernelManager] 初始化完成，核目录: {self.kernel_dir}")
            print(f"[SPICEKernelManager] 可用核文件: {len(self._kernels)} 个")
    
    def _load_kernel_database(self):
        """加载核文件数据库"""
        for kernel_id, kernel_data in self.COMMON_KERNELS.items():
            local_path = self.kernel_dir / kernel_data['name']
            
            # 检查是否有解压版本
            if kernel_data.get('compressed', False):
                uncompressed_path = local_path.with_suffix('')  # 移除.gz后缀
                if uncompressed_path.exists():
                    local_path = uncompressed_path
            
            self._kernels[kernel_id] = KernelInfo(
                name=kernel_data['name'],
                category=kernel_data['category'],
                url=kernel_data['url'],
                local_path=local_path,
                description=kernel_data.get('description', ''),
                required=kernel_data.get('required', True),
                size=kernel_data.get('size', 0),
                md5=kernel_data.get('md5'),
                compressed=kernel_data.get('compressed', False)
            )
    
    def _load_state(self):
        """加载状态信息"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                for kernel_id, kernel in self._kernels.items():
                    if kernel_id in state_data:
                        state = state_data[kernel_id]
                        if 'last_update' in state and state['last_update']:
                            try:
                                kernel.last_update = datetime.fromisoformat(state['last_update'])
                            except ValueError:
                                # 尝试其他格式
                                try:
                                    kernel.last_update = datetime.strptime(
                                        state['last_update'], '%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    kernel.last_update = None
                        if 'size' in state:
                            kernel.size = state['size']
                        if 'md5' in state:
                            kernel.md5 = state['md5']
            except Exception as e:
                if self.verbose:
                    print(f"[SPICEKernelManager] 加载状态失败: {e}")
    
    def _save_state(self):
        """保存状态信息"""
        state_data = {}
        for kernel_id, kernel in self._kernels.items():
            state_data[kernel_id] = {
                'name': kernel.name,
                'last_update': kernel.last_update.isoformat() if kernel.last_update else None,
                'size': kernel.size,
                'md5': kernel.md5,
                'category': kernel.category,
                'description': kernel.description
            }
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.verbose:
                print(f"[SPICEKernelManager] 保存状态失败: {e}")
    
    def list_kernels(self) -> List[Dict]:
        """列出所有可用的核文件"""
        with self._lock:
            kernels_info = []
            for kernel_id, kernel in self._kernels.items():
                exists = kernel.local_path.exists()
                size_mb = kernel.size / (1024*1024) if kernel.size > 0 else 0
                
                if exists and kernel.size == 0:
                    # 获取实际文件大小
                    try:
                        kernel.size = kernel.local_path.stat().st_size
                        size_mb = kernel.size / (1024*1024)
                    except OSError:
                        pass
                
                kernels_info.append({
                    'id': kernel_id,
                    'name': kernel.name,
                    'category': kernel.category,
                    'required': kernel.required,
                    'exists': exists,
                    'local_path': str(kernel.local_path),
                    'size_mb': f"{size_mb:.2f}" if size_mb > 0 else "未知",
                    'last_update': kernel.last_update.isoformat() if kernel.last_update else "从未",
                    'description': kernel.description
                })
            
            return kernels_info
    
    def get_kernel(self, kernel_id: str) -> Optional[KernelInfo]:
        """获取特定核文件信息"""
        with self._lock:
            return self._kernels.get(kernel_id)
    
    def add_kernel(self, kernel_id: str, name: str, url: str, 
                   category: str = "generic", description: str = "", 
                   required: bool = False, compressed: bool = False):
        """
        添加自定义核文件
        
        Args:
            kernel_id: 核标识符
            name: 文件名
            url: 下载URL
            category: 类别
            description: 描述
            required: 是否必需
            compressed: 是否为压缩文件
        """
        with self._lock:
            local_path = self.kernel_dir / name
            
            self._kernels[kernel_id] = KernelInfo(
                name=name,
                category=category,
                url=url,
                local_path=local_path,
                description=description,
                required=required,
                compressed=compressed
            )
            
            if self.verbose:
                print(f"[SPICEKernelManager] 添加核文件: {kernel_id} -> {name}")
    
    def remove_kernel(self, kernel_id: str, delete_file: bool = False):
        """
        移除核文件
        
        Args:
            kernel_id: 核标识符
            delete_file: 是否删除本地文件
        """
        with self._lock:
            if kernel_id in self._kernels:
                kernel = self._kernels[kernel_id]
                
                if delete_file and kernel.local_path.exists():
                    try:
                        kernel.local_path.unlink()
                        if self.verbose:
                            print(f"[SPICEKernelManager] 删除文件: {kernel.local_path}")
                    except Exception as e:
                        print(f"[SPICEKernelManager] 删除文件失败: {e}")
                
                del self._kernels[kernel_id]
                
                if self.verbose:
                    print(f"[SPICEKernelManager] 移除核文件: {kernel_id}")
    
    def _download_file(self, url: str, dest_path: Path, kernel_name: str) -> bool:
        """下载单个文件"""
        try:
            # 创建临时文件
            temp_file = dest_path.with_suffix('.downloading')
            
            # 使用requests下载
            headers = {'User-Agent': 'MCPC-SPICE-Kernel-Manager/1.0'}
            response = requests.get(url, stream=True, timeout=self.config.timeout, headers=headers)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 显示进度条
            with open(temp_file, 'wb') as f:
                if self.verbose:
                    with tqdm(
                        desc=f"下载 {kernel_name}",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            # 验证文件大小
            if total_size > 0:
                actual_size = temp_file.stat().st_size
                if actual_size != total_size:
                    print(f"[SPICEKernelManager] 警告: 文件大小不匹配 "
                          f"({actual_size} != {total_size})")
                    return False
            
            # 移动临时文件到目标位置
            shutil.move(temp_file, dest_path)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"[SPICEKernelManager] 下载失败 {url}: {e}")
            # 清理临时文件
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()
            return False
    
    def _decompress_file(self, compressed_path: Path, dest_path: Path) -> bool:
        """解压文件"""
        try:
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            print(f"[SPICEKernelManager] 解压失败 {compressed_path}: {e}")
            return False
    
    def download_kernel(self, kernel_id: str, force: bool = False) -> bool:
        """
        下载单个核文件
        
        Args:
            kernel_id: 核标识符
            force: 强制重新下载（即使已存在）
            
        Returns:
            bool: 下载是否成功
        """
        with self._lock:
            if kernel_id not in self._kernels:
                print(f"[SPICEKernelManager] 错误: 核文件 '{kernel_id}' 不存在")
                return False
            
            if not HAS_DEPENDENCIES:
                print("[SPICEKernelManager] 错误: 缺少requests库，无法下载")
                return False
            
            kernel = self._kernels[kernel_id]
            
            # 检查文件是否已存在
            if kernel.local_path.exists() and not force:
                if self.verbose:
                    print(f"[SPICEKernelManager] 文件已存在: {kernel.local_path}")
                
                # 更新文件大小信息
                if kernel.size == 0:
                    kernel.size = kernel.local_path.stat().st_size
                
                return True
            
            # 下载文件
            if self.verbose:
                print(f"[SPICEKernelManager] 开始下载: {kernel.name} ({kernel_id})")
                print(f"  URL: {kernel.url}")
            
            success = False
            for attempt in range(self.config.max_retries):
                if attempt > 0:
                    print(f"[SPICEKernelManager] 重试 {attempt}/{self.config.max_retries}...")
                    time.sleep(2 ** attempt)  # 指数退避
                
                if kernel.compressed:
                    # 下载压缩文件
                    compressed_path = kernel.local_path.with_suffix('.gz')
                    if self._download_file(kernel.url, compressed_path, kernel.name):
                        # 解压文件
                        if self._decompress_file(compressed_path, kernel.local_path):
                            compressed_path.unlink()  # 删除压缩文件
                            success = True
                            break
                else:
                    # 直接下载
                    if self._download_file(kernel.url, kernel.local_path, kernel.name):
                        success = True
                        break
            
            if success:
                # 更新状态
                kernel.last_update = datetime.now()
                kernel.size = kernel.local_path.stat().st_size
                self._save_state()
                
                if self.verbose:
                    size_mb = kernel.size / (1024*1024)
                    print(f"[SPICEKernelManager] 下载完成: {kernel.local_path} "
                          f"({size_mb:.2f} MB)")
                
                return True
            else:
                print(f"[SPICEKernelManager] 下载失败，已重试 {self.config.max_retries} 次")
                return False
    
    def download_kernels(self, kernel_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """
        批量下载核文件
        
        Args:
            kernel_ids: 核文件ID列表
            force: 强制重新下载
            
        Returns:
            Dict[str, bool]: 下载结果 {核ID: 是否成功}
        """
        if not HAS_DEPENDENCIES:
            return {kernel_id: False for kernel_id in kernel_ids}
        
        results = {}
        
        if self.verbose:
            print(f"[SPICEKernelManager] 开始批量下载 {len(kernel_ids)} 个文件")
        
        # 使用线程池并行下载
        with ThreadPoolExecutor(max_workers=self.config.parallel_downloads) as executor:
            future_to_kernel = {
                executor.submit(self.download_kernel, kernel_id, force): kernel_id
                for kernel_id in kernel_ids
            }
            
            for future in as_completed(future_to_kernel):
                kernel_id = future_to_kernel[future]
                try:
                    results[kernel_id] = future.result()
                except Exception as e:
                    print(f"[SPICEKernelManager] 下载 {kernel_id} 时出错: {e}")
                    results[kernel_id] = False
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r)
        
        if self.verbose:
            print(f"[SPICEKernelManager] 批量下载完成: {success_count}/{len(kernel_ids)} 成功")
        
        return results
    
    def download_all(self, required_only: bool = True, force: bool = False) -> Dict[str, bool]:
        """
        下载所有核文件
        
        Args:
            required_only: 仅下载必需的核文件
            force: 强制重新下载
            
        Returns:
            Dict[str, bool]: 下载结果 {核ID: 是否成功}
        """
        kernel_ids = []
        for kernel_id, kernel in self._kernels.items():
            if required_only and not kernel.required:
                continue
            kernel_ids.append(kernel_id)
        
        return self.download_kernels(kernel_ids, force=force)
    
    def check_updates(self) -> Dict[str, bool]:
        """
        检查核文件更新
        
        Returns:
            Dict[str, bool]: 更新状态 {核ID: 是否需要更新}
        """
        if not HAS_DEPENDENCIES:
            return {}
        
        updates_needed = {}
        
        for kernel_id, kernel in self._kernels.items():
            # 如果文件不存在，需要下载
            if not kernel.local_path.exists():
                updates_needed[kernel_id] = True
                continue
            
            # 检查是否需要更新（基于时间间隔）
            if kernel.last_update and self.config.check_updates:
                days_since_update = (datetime.now() - kernel.last_update).days
                if days_since_update >= self.config.update_interval_days:
                    updates_needed[kernel_id] = True
                else:
                    updates_needed[kernel_id] = False
            else:
                updates_needed[kernel_id] = False
        
        return updates_needed
    
    def verify_kernel(self, kernel_id: str) -> Tuple[bool, Optional[str]]:
        """
        验证核文件完整性
        
        Args:
            kernel_id: 核标识符
            
        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        if kernel_id not in self._kernels:
            return False, f"核文件 '{kernel_id}' 不存在"
        
        kernel = self._kernels[kernel_id]
        
        # 检查文件是否存在
        if not kernel.local_path.exists():
            return False, f"文件不存在: {kernel.local_path}"
        
        # 检查文件大小
        file_size = kernel.local_path.stat().st_size
        if kernel.size > 0 and abs(file_size - kernel.size) > 1024:  # 允许1KB误差
            return False, f"文件大小不匹配: {file_size} != {kernel.size} (相差 {abs(file_size - kernel.size)} 字节)"
        
        # 检查MD5校验和（如果有）
        if kernel.md5:
            md5_hash = hashlib.md5()
            with open(kernel.local_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            
            actual_md5 = md5_hash.hexdigest()
            if actual_md5 != kernel.md5:
                return False, f"MD5校验失败: {actual_md5} != {kernel.md5}"
        
        return True, None
    
    def setup_for_mission(self, mission_type: str = "earth_moon") -> List[str]:
        """
        为特定任务设置核文件
        
        Args:
            mission_type: 任务类型 ('earth_moon', 'deep_space', 'mars', 'custom')
            
        Returns:
            List[str]: 成功下载的核文件列表
        """
        mission_kernels = {
            'earth_moon': ['de440', 'pck00010', 'naif0012', 'earth_200101', 'moon_pa'],
            'deep_space': ['de440', 'pck00010', 'naif0012'],
            'mars': ['de440', 'pck00010', 'naif0012'],
            'custom': ['de440', 'pck00010', 'naif0012'],
            'lightweight': ['de440_small', 'pck00010', 'naif0012']
        }
        
        if mission_type not in mission_kernels:
            print(f"[SPICEKernelManager] 警告: 未知任务类型 '{mission_type}'，使用默认")
            mission_type = 'custom'
        
        kernel_ids = mission_kernels[mission_type]
        
        if self.verbose:
            print(f"[SPICEKernelManager] 为 {mission_type} 任务设置核文件")
        
        # 下载所需的核文件
        results = self.download_kernels(kernel_ids)
        
        successful = [kernel_id for kernel_id, success in results.items() if success]
        
        return successful
    
    def get_kernel_paths(self, mission_type: str = "earth_moon") -> List[str]:
        """
        获取特定任务所需的核文件路径
        
        Args:
            mission_type: 任务类型
            
        Returns:
            List[str]: 核文件路径列表
        """
        mission_kernels = {
            'earth_moon': ['de440', 'pck00010', 'naif0012', 'earth_200101', 'moon_pa'],
            'deep_space': ['de440', 'pck00010', 'naif0012'],
            'mars': ['de440', 'pck00010', 'naif0012'],
            'custom': ['de440', 'pck00010', 'naif0012'],
            'lightweight': ['de440_small', 'pck00010', 'naif0012']
        }
        
        if mission_type not in mission_kernels:
            mission_type = 'custom'
        
        kernel_ids = mission_kernels[mission_type]
        paths = []
        
        for kernel_id in kernel_ids:
            if kernel_id in self._kernels:
                kernel = self._kernels[kernel_id]
                if kernel.local_path.exists():
                    paths.append(str(kernel.local_path))
                else:
                    if self.verbose:
                        print(f"[SPICEKernelManager] 警告: 核文件不存在 {kernel_id}")
        
        return paths
    
    def clean_cache(self, days_old: int = 30, dry_run: bool = False) -> List[str]:
        """
        清理旧的缓存文件
        
        Args:
            days_old: 删除超过指定天数的文件
            dry_run: 仅显示要删除的文件，不实际删除
            
        Returns:
            List[str]: 删除的文件列表
        """
        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_files = []
        
        for file_path in self.kernel_dir.glob("*"):
            if file_path.is_file() and file_path.name != "kernel_state.json":
                # 获取文件修改时间
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        deleted_files.append(str(file_path))
                        
                        if not dry_run:
                            try:
                                file_path.unlink()
                                if self.verbose:
                                    print(f"[SPICEKernelManager] 删除旧文件: {file_path}")
                            except Exception as e:
                                print(f"[SPICEKernelManager] 删除文件失败 {file_path}: {e}")
                except OSError:
                    pass
        
        return deleted_files
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_kernels = len(self._kernels)
            downloaded = sum(1 for k in self._kernels.values() if k.local_path.exists())
            total_size = sum(k.size for k in self._kernels.values() if k.local_path.exists())
            total_size_mb = total_size / (1024*1024)
            
            return {
                'total_kernels': total_kernels,
                'downloaded': downloaded,
                'total_size_mb': f"{total_size_mb:.2f}",
                'kernel_dir': str(self.kernel_dir)
            }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"SPICEKernelManager(kernels={stats['downloaded']}/{stats['total_kernels']}, "
                f"size={stats['total_size_mb']} MB, directory={self.kernel_dir})")


# 命令行接口
def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SPICE核文件管理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --list
  %(prog)s --download
  %(prog)s --mission earth_moon
  %(prog)s --check-updates
  %(prog)s --clean 30
        """
    )
    
    parser.add_argument("--list", action="store_true", help="列出所有核文件")
    parser.add_argument("--download", nargs="*", metavar="KERNEL_ID", 
                       help="下载指定核文件（不指定则下载所有必需文件）")
    parser.add_argument("--mission", choices=["earth_moon", "deep_space", "mars", "custom", "lightweight"], 
                       help="为特定任务设置核文件")
    parser.add_argument("--check-updates", action="store_true", help="检查更新")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--clean", type=int, nargs="?", metavar="DAYS", const=30, 
                       help="清理超过指定天数的旧文件（默认30天）")
    parser.add_argument("--dir", default="~/.mission_sim/spice_kernels", 
                       help="核文件目录（默认: ~/.mission_sim/spice_kernels）")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    
    args = parser.parse_args()
    
    # 创建管理器
    config = KernelConfig(
        kernel_dir=Path(args.dir).expanduser(),
        verbose=args.verbose
    )
    
    manager = SPICEKernelManager(config)
    
    # 执行命令
    if args.list:
        kernels = manager.list_kernels()
        print(f"\nSPICE核文件列表 ({len(kernels)} 个):")
        print("=" * 100)
        print(f"{'ID':20} {'名称':25} {'类别':12} {'大小':>8} {'状态':6} {'最后更新':20} {'描述'}")
        print("-" * 100)
        for k in kernels:
            status = "✓" if k['exists'] else "✗"
            size_display = k['size_mb'] if k['size_mb'] != "未知" else "  N/A  "
            print(f"{k['id']:20} {k['name']:25} {k['category']:12} {size_display:>8} MB "
                  f"{status:6} {k['last_update']:20} {k['description']}")
        print("=" * 100)
    
    elif args.download is not None:
        if len(args.download) == 0:
            # 下载所有必需文件
            results = manager.download_all(required_only=True, force=args.force)
            print(f"\n下载结果:")
            for kernel_id, success in results.items():
                status = "✓ 成功" if success else "✗ 失败"
                print(f"  {kernel_id}: {status}")
        else:
            # 下载指定文件
            results = manager.download_kernels(args.download, force=args.force)
            print(f"\n下载结果:")
            for kernel_id, success in results.items():
                status = "✓ 成功" if success else "✗ 失败"
                print(f"  {kernel_id}: {status}")
    
    elif args.mission:
        print(f"为 {args.mission} 任务设置核文件...")
        successful = manager.setup_for_mission(args.mission)
        print(f"成功下载 {len(successful)} 个核文件: {', '.join(successful)}")
    
    elif args.check_updates:
        updates = manager.check_updates()
        need_update = [k for k, v in updates.items() if v]
        
        if need_update:
            print(f"需要更新的核文件 ({len(need_update)} 个):")
            for kernel_id in need_update:
                kernel = manager.get_kernel(kernel_id)
                last_update = kernel.last_update.strftime('%Y-%m-%d') if kernel.last_update else "从未"
                print(f"  - {kernel_id}: {kernel.name} (最后更新: {last_update})")
            
            response = input("\n是否立即更新？ (y/N): ").strip().lower()
            if response == 'y':
                for kernel_id in need_update:
                    success = manager.download_kernel(kernel_id, force=True)
                    status = "成功" if success else "失败"
                    print(f"  {kernel_id}: {status}")
        else:
            print("所有核文件都是最新的")
    
    elif args.clean is not None:
        print(f"清理超过 {args.clean} 天的旧文件...")
        if args.verbose:
            print("将删除以下文件:")
            manager.clean_cache(days_old=args.clean, dry_run=True)
        
        response = input("确认删除？ (y/N): ").strip().lower()
        if response == 'y':
            deleted = manager.clean_cache(days_old=args.clean, dry_run=False)
            if deleted:
                print(f"\n删除了 {len(deleted)} 个文件:")
                for f in deleted:
                    print(f"  - {Path(f).name}")
            else:
                print("没有需要清理的文件")
        else:
            print("取消清理")
    
    elif args.stats:
        stats = manager.get_stats()
        print(f"\nSPICE核文件统计:")
        print(f"  核目录: {stats['kernel_dir']}")
        print(f"  总核文件: {stats['total_kernels']}")
        print(f"  已下载: {stats['downloaded']}")
        print(f"  总大小: {stats['total_size_mb']} MB")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
