"""
SPICE Kernel Manager

Automated download, update, and management of SPICE kernel files (for high-precision ephemeris calculations).
Supports DE440 planetary ephemerides, attitude files, timing files, etc.

Features:
1. Automatic download of missing kernel files
2. Version management and update checking
3. Local caching to avoid repeated downloads
4. Multi-threaded download support
5. File integrity verification (MD5 checksum)
6. HTTP proxy support for restricted networks

Proxy Usage Examples:
  1. Command line: --proxy http://proxy.example.com:8080
  2. Environment variable: export HTTP_PROXY=http://proxy.example.com:8080
  3. With authentication: --proxy http://username:password@proxy.example.com:8080

Common Issues & Solutions:
  1. Connection timeout: Increase timeout with --timeout 300
  2. Proxy required: Use --proxy option or set HTTP_PROXY environment variable
  3. Slow downloads: Use --verbose to monitor progress, consider using a faster proxy

Author: MCPC Development Team
Version: 1.1.0
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
    print("Warning: Missing requests or tqdm library, automatic download unavailable")
    print("Please install: pip install requests tqdm")

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
    proxy: Optional[str] = None    # HTTP代理 (例如: http://proxy.example.com:8080)
    proxy_auth: Optional[Tuple[str, str]] = None  # 代理认证 (用户名, 密码)


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
            'description': 'DE440 Planetary Ephemeris (1550-2650) - Well-established standard',
            'required': True,  # 保持默认
            'size': 135_000_000,  # 约135MB
            'compressed': False
        },
        'de442': {
            'name': 'de442.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp',
            'description': 'DE442 Planetary Ephemeris (1549-2650) - Updated Uranus barycenter with occultation data and extended Mars/Juno ranging',
            'required': False,  # 作为可选升级
            'size': 150_000_000,  # 约150MB
            'compressed': False
        },
        'de440_small': {
            'name': 'de440s.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp',
            'description': 'DE440 Simplified Planetary Ephemeris (1550-2650) - For storage-constrained applications',
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
        Initialize SPICE kernel manager
        
        Args:
            config: Kernel configuration, uses default if None
        """
        self.config = config or KernelConfig()
        self.kernel_dir = self.config.kernel_dir
        
        # Set verbose attribute early, before it's used
        self.verbose = self.config.verbose
        
        # 检查环境变量中的代理设置
        if not self.config.proxy:
            # 检查常见的环境变量
            for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                if env_var in os.environ:
                    self.config.proxy = os.environ[env_var]
                    if self.verbose:
                        print(f"[SPICEKernelManager] Using proxy from environment variable {env_var}: {self.config.proxy}")
                    break
    
        # Ensure directory exists
        self.kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # Load kernel database
        self._kernels: Dict[str, KernelInfo] = {}
        self._load_kernel_database()
        
        # State file path
        self.state_file = self.kernel_dir / "kernel_state.json"
        self._load_state()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        if not HAS_DEPENDENCIES:
            if self.config.auto_download:
                warnings.warn("Missing requests or tqdm library, automatic download unavailable")
        
        if self.verbose:
            print(f"[SPICEKernelManager] Initialization complete. Kernel directory: {self.kernel_dir}")
            print(f"[SPICEKernelManager] Available kernels: {len(self._kernels)}")
    
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
        """Load state information"""
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
                                # Try alternative format
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
                    print(f"[SPICEKernelManager] Failed to load state: {e}")
    
    def _save_state(self):
        """Save state information"""
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
                print(f"[SPICEKernelManager] Failed to save state: {e}")
    
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
        Add custom kernel file
        
        Args:
            kernel_id: Kernel identifier
            name: Filename
            url: Download URL
            category: Category
            description: Description
            required: Whether required
            compressed: Whether compressed file
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
                print(f"[SPICEKernelManager] Added kernel: {kernel_id} -> {name}")
    
    def remove_kernel(self, kernel_id: str, delete_file: bool = False):
        """
        Remove kernel file
        
        Args:
            kernel_id: Kernel identifier
            delete_file: Whether to delete local file
        """
        with self._lock:
            if kernel_id in self._kernels:
                kernel = self._kernels[kernel_id]
                
                if delete_file and kernel.local_path.exists():
                    try:
                        kernel.local_path.unlink()
                        if self.verbose:
                            print(f"[SPICEKernelManager] Deleted file: {kernel.local_path}")
                    except Exception as e:
                        print(f"[SPICEKernelManager] Failed to delete file: {e}")
                
                del self._kernels[kernel_id]
                
                if self.verbose:
                    print(f"[SPICEKernelManager] Removed kernel: {kernel_id}")
    
    def _download_file(self, url: str, dest_path: Path, kernel_name: str) -> bool:
        """Download single file"""
        try:
            # Create temporary file
            temp_file = dest_path.with_suffix('.downloading')
            
            # 配置请求参数
            headers = {'User-Agent': 'MCPC-SPICE-Kernel-Manager/1.0'}
            
            # 配置代理
            proxies = None
            if self.config.proxy:
                proxies = {
                    'http': self.config.proxy,
                    'https': self.config.proxy
                }
                if self.verbose:
                    print(f"[SPICEKernelManager] Using proxy: {self.config.proxy}")
        
            # 配置认证
            auth = None
            if self.config.proxy_auth:
                auth = self.config.proxy_auth
            
            # Download with requests
            response = requests.get(
                url, 
                stream=True, 
                timeout=self.config.timeout, 
                headers=headers,
                proxies=proxies,
                auth=auth
            )
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Show progress bar
            with open(temp_file, 'wb') as f:
                if self.verbose:
                    with tqdm(
                        desc=f"Downloading {kernel_name}",
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
            
            # Verify file size
            if total_size > 0:
                actual_size = temp_file.stat().st_size
                if actual_size != total_size:
                    print(f"[SPICEKernelManager] Warning: File size mismatch "
                          f"({actual_size} != {total_size})")
                    return False
            
            # Move temporary file to destination
            shutil.move(temp_file, dest_path)
            
            return True
            
        except requests.exceptions.ProxyError as e:
            print(f"[SPICEKernelManager] Proxy error: {e}")
            print("Please check your proxy configuration or try without proxy")
            return False
        except requests.exceptions.ConnectTimeout as e:
            print(f"[SPICEKernelManager] Connection timeout: {e}")
            print("Consider increasing timeout with --timeout option or using a proxy")
            return False
        except requests.exceptions.RequestException as e:
            print(f"[SPICEKernelManager] Download failed {url}: {e}")
            # Clean up temporary file
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
            print(f"[SPICEKernelManager] Decompression failed {compressed_path}: {e}")
            return False
    
    def download_kernel(self, kernel_id: str, force: bool = False) -> bool:
        """
        Download single kernel file
        
        Args:
            kernel_id: Kernel identifier
            force: Force re-download (even if exists)
            
        Returns:
            bool: Whether download succeeded
        """
        with self._lock:
            if kernel_id not in self._kernels:
                print(f"[SPICEKernelManager] Error: Kernel '{kernel_id}' not found")
                return False
            
            if not HAS_DEPENDENCIES:
                print("[SPICEKernelManager] Error: Missing requests library, cannot download")
                return False
            
            # 检查网络连接
            if not self.test_connection():
                print("[SPICEKernelManager] Error: Cannot connect to NASA servers")
                print("Possible solutions:")
                print("  1. Check your internet connection")
                print("  2. Use --proxy option to set up HTTP proxy")
                print("  3. Increase timeout with --timeout option (default: 30s)")
                print("  4. Try again later - NASA servers may be temporarily unavailable")
                return False
            
            kernel = self._kernels[kernel_id]
            
            # Check if file already exists
            if kernel.local_path.exists() and not force:
                if self.verbose:
                    print(f"[SPICEKernelManager] File already exists: {kernel.local_path}")
                
                # Update file size information
                if kernel.size == 0:
                    kernel.size = kernel.local_path.stat().st_size
                
                return True
            
            # Download file
            if self.verbose:
                print(f"[SPICEKernelManager] Starting download: {kernel.name} ({kernel_id})")
                print(f"  URL: {kernel.url}")
                if self.config.proxy:
                    print(f"  Proxy: {self.config.proxy}")
            
            success = False
            for attempt in range(self.config.max_retries):
                if attempt > 0:
                    print(f"[SPICEKernelManager] Retry {attempt}/{self.config.max_retries}...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                if kernel.compressed:
                    # Download compressed file
                    compressed_path = kernel.local_path.with_suffix('.gz')
                    if self._download_file(kernel.url, compressed_path, kernel.name):
                        # Decompress file
                        if self._decompress_file(compressed_path, kernel.local_path):
                            compressed_path.unlink()  # Delete compressed file
                            success = True
                            break
                else:
                    # Direct download
                    if self._download_file(kernel.url, kernel.local_path, kernel.name):
                        success = True
                        break
            
            if success:
                # Update state
                kernel.last_update = datetime.now()
                kernel.size = kernel.local_path.stat().st_size
                self._save_state()
                
                if self.verbose:
                    size_mb = kernel.size / (1024*1024)
                    print(f"[SPICEKernelManager] Download complete: {kernel.local_path} "
                          f"({size_mb:.2f} MB)")
                
                return True
            else:
                print(f"[SPICEKernelManager] Download failed after {self.config.max_retries} retries")
                print("\nTroubleshooting tips:")
                print("  1. Check if URL is accessible in browser: " + kernel.url)
                print("  2. Try with --proxy option: --proxy http://your-proxy:port")
                print("  3. Increase timeout: --timeout 300")
                print("  4. Use --verbose for detailed logs")
                print("  5. Check firewall/proxy settings")
                return False
    
    def download_kernels(self, kernel_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """
        Batch download kernel files
        
        Args:
            kernel_ids: List of kernel IDs
            force: Force re-download
            
        Returns:
            Dict[str, bool]: Download results {kernel_id: success}
        """
        if not HAS_DEPENDENCIES:
            return {kernel_id: False for kernel_id in kernel_ids}
        
        results = {}
        
        if self.verbose:
            print(f"[SPICEKernelManager] Starting batch download of {len(kernel_ids)} files")
        
        # Use thread pool for parallel downloads
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
                    print(f"[SPICEKernelManager] Error downloading {kernel_id}: {e}")
                    results[kernel_id] = False
        
        # Count results
        success_count = sum(1 for r in results.values() if r)
        
        if self.verbose:
            print(f"[SPICEKernelManager] Batch download complete: {success_count}/{len(kernel_ids)} succeeded")
        
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
        Verify kernel file integrity
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Tuple[bool, Optional[str]]: (valid, error_message)
        """
        if kernel_id not in self._kernels:
            return False, f"Kernel '{kernel_id}' not found"
        
        kernel = self._kernels[kernel_id]
        
        # Check if file exists
        if not kernel.local_path.exists():
            return False, f"File not found: {kernel.local_path}"
        
        # Check file size
        file_size = kernel.local_path.stat().st_size
        if kernel.size > 0 and abs(file_size - kernel.size) > 1024:  # Allow 1KB difference
            return False, f"File size mismatch: {file_size} != {kernel.size} (difference: {abs(file_size - kernel.size)} bytes)"
        
        # Check MD5 checksum (if available)
        if kernel.md5:
            md5_hash = hashlib.md5()
            with open(kernel.local_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            
            actual_md5 = md5_hash.hexdigest()
            if actual_md5 != kernel.md5:
                return False, f"MD5 checksum failed: {actual_md5} != {kernel.md5}"
        
        return True, None
    
    def setup_for_mission(self, mission_type: str = "earth_moon", 
                         ephemeris: str = "de440") -> List[str]:
        """
        设置特定任务的内核文件
        
        Args:
            mission_type: 任务类型 ('earth_moon', 'deep_space', 'mars', 'custom', 'lightweight')
            ephemeris: 星历版本选择：
                - 'de440': 【默认】DE440 (1550-2650) - 经过充分验证的稳定版本
                - 'de442': DE442 (1549-2650) - 最新版本，特别改进天王星轨道和外行星数据
                - 'de440_small': DE440简化版 (1550-2650) - 存储受限时使用
        
        Returns:
            List[str]: 成功下载的内核文件ID列表
        
        建议：
            - 新任务或外行星任务：考虑使用DE442
            - 地球/月球任务：DE440已足够精确
            - 天王星相关任务：必须使用DE442
            - 存储受限环境：使用de440_small
        """
        # 验证星历版本选择
        valid_ephemeris = ['de440', 'de442', 'de440_small']
        if ephemeris not in valid_ephemeris:
            print(f"[SPICEKernelManager] Warning: Unknown ephemeris version '{ephemeris}', using 'de440'")
            ephemeris = 'de440'
        
        # 根据任务类型和星历版本选择内核
        mission_kernels = {
            'earth_moon': [ephemeris, 'pck00010', 'naif0012', 'earth_200101', 'moon_pa'],
            'deep_space': [ephemeris, 'pck00010', 'naif0012'],
            'mars': [ephemeris, 'pck00010', 'naif0012'],
            'custom': [ephemeris, 'pck00010', 'naif0012'],
            'lightweight': ['de440_small', 'pck00010', 'naif0012']  # lightweight总是使用de440_small
        }
        
        if mission_type not in mission_kernels:
            print(f"[SPICEKernelManager] Warning: Unknown mission type '{mission_type}', using 'custom'")
            mission_type = 'custom'
        
        kernel_ids = mission_kernels[mission_type]
        
        if self.verbose:
            print(f"[SPICEKernelManager] Setting up kernel files for {mission_type} mission")
            print(f"[SPICEKernelManager] Using ephemeris: {ephemeris}")
            
            # 提供选择建议
            if ephemeris == 'de442':
                print("[SPICEKernelManager] Note: DE442 selected - includes updated Uranus barycenter and extended Mars/Juno ranging data")
            elif ephemeris == 'de440_small':
                print("[SPICEKernelManager] Note: DE440 small version selected - reduced size for storage-constrained applications")
        
        # 下载所需内核文件
        results = self.download_kernels(kernel_ids)
        
        successful = [kernel_id for kernel_id, success in results.items() if success]
        
        return successful
    
    def get_kernel_paths(self, mission_type: str = "earth_moon", 
                        ephemeris: str = "de440") -> List[str]:
        """
        获取特定任务的内核文件路径
        
        Args:
            mission_type: 任务类型 ('earth_moon', 'deep_space', 'mars', 'custom', 'lightweight')
            ephemeris: 星历版本选择 ('de440', 'de442', 'de440_small')
            
        Returns:
            List[str]: 内核文件路径列表
        """
        # 验证星历版本选择
        valid_ephemeris = ['de440', 'de442', 'de440_small']
        if ephemeris not in valid_ephemeris:
            print(f"[SPICEKernelManager] Warning: Unknown ephemeris version '{ephemeris}', using 'de440'")
            ephemeris = 'de440'
        
        # 根据任务类型和星历版本选择内核
        mission_kernels = {
            'earth_moon': [ephemeris, 'pck00010', 'naif0012', 'earth_200101', 'moon_pa'],
            'deep_space': [ephemeris, 'pck00010', 'naif0012'],
            'mars': [ephemeris, 'pck00010', 'naif0012'],
            'custom': [ephemeris, 'pck00010', 'naif0012'],
            'lightweight': ['de440_small', 'pck00010', 'naif0012']  # lightweight总是使用de440_small
        }
        
        if mission_type not in mission_kernels:
            print(f"[SPICEKernelManager] Warning: Unknown mission type '{mission_type}', using 'custom'")
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
                        print(f"[SPICEKernelManager] Warning: Kernel file not found: {kernel_id}")
        
        return paths
    
    def clean_cache(self, days_old: int = 30, dry_run: bool = False) -> List[str]:
        """
        Clean old cache files
        
        Args:
            days_old: Delete files older than specified days
            dry_run: Only show files to delete, don't actually delete
            
        Returns:
            List[str]: List of deleted files
        """
        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_files = []
        
        for file_path in self.kernel_dir.glob("*"):
            if file_path.is_file() and file_path.name != "kernel_state.json":
                # Get file modification time
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        deleted_files.append(str(file_path))
                        
                        if not dry_run:
                            try:
                                file_path.unlink()
                                if self.verbose:
                                    print(f"[SPICEKernelManager] Deleted old file: {file_path}")
                            except Exception as e:
                                print(f"[SPICEKernelManager] Failed to delete file {file_path}: {e}")
                except OSError:
                    pass
        
        return deleted_files
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
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
    
    def test_connection(self, url: str = "https://naif.jpl.nasa.gov/pub/naif/", timeout: int = 10) -> bool:
        """
        测试网络连接
        
        Args:
            url: 测试URL
            timeout: 超时时间
            
        Returns:
            bool: 是否连接成功
        """
        try:
            if self.verbose:
                print(f"[SPICEKernelManager] Testing connection to {url}...")
            
            headers = {'User-Agent': 'MCPC-SPICE-Kernel-Manager/1.0'}
            proxies = None
            if self.config.proxy:
                proxies = {
                    'http': self.config.proxy,
                    'https': self.config.proxy
                }
            
            response = requests.get(url, timeout=timeout, headers=headers, proxies=proxies)
            
            if response.status_code == 200:
                if self.verbose:
                    print("[SPICEKernelManager] Connection test successful")
                return True
            else:
                print(f"[SPICEKernelManager] Connection test failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[SPICEKernelManager] Connection test failed: {e}")
            return False

    def __repr__(self):
        stats = self.get_stats()
        return (f"SPICEKernelManager(kernels={stats['downloaded']}/{stats['total_kernels']}, "
                f"size={stats['total_size_mb']} MB, directory={self.kernel_dir})")


# 命令行接口
def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SPICE Kernel Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --download
  %(prog)s --mission earth_moon
  %(prog)s --check-updates
  %(prog)s --clean 30
  %(prog)s --proxy http://proxy.example.com:8080
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all kernel files")
    parser.add_argument("--download", nargs="*", metavar="KERNEL_ID", 
                       help="Download specified kernel files (download all required if not specified)")
    parser.add_argument("--mission", choices=["earth_moon", "deep_space", "mars", "custom", "lightweight"], 
                       help="Set up kernel files for specific mission")
    parser.add_argument("--ephemeris", choices=["de440", "de442", "de440_small"], default="de440",
                       help="Ephemeris version (default: de440)")
    parser.add_argument("--check-updates", action="store_true", help="Check for updates")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--clean", type=int, nargs="?", metavar="DAYS", const=30, 
                       help="Clean files older than specified days (default: 30)")
    parser.add_argument("--dir", default="~/.mission_sim/spice_kernels", 
                       help="Kernel directory (default: ~/.mission_sim/spice_kernels)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--proxy", help="HTTP proxy (e.g., http://proxy.example.com:8080)")
    parser.add_argument("--timeout", type=int, default=30, 
                       help="Download timeout in seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Create manager
    config = KernelConfig(
        kernel_dir=Path(args.dir).expanduser(),
        verbose=args.verbose,
        proxy=args.proxy,
        timeout=args.timeout
    )
    
    manager = SPICEKernelManager(config)
    
    # Execute commands
    if args.list:
        kernels = manager.list_kernels()
        print(f"\nSPICE Kernel File List ({len(kernels)}):")
        print("=" * 100)
        print(f"{'ID':20} {'Name':25} {'Category':12} {'Size':>8} {'Status':6} {'Last Update':20} {'Description'}")
        print("-" * 100)
        for k in kernels:
            status = "✓" if k['exists'] else "✗"
            size_display = k['size_mb'] if k['size_mb'] != "unknown" else "  N/A  "
            print(f"{k['id']:20} {k['name']:25} {k['category']:12} {size_display:>8} MB "
                  f"{status:6} {k['last_update']:20} {k['description']}")
        print("=" * 100)
    
    elif args.download is not None:
        if len(args.download) == 0:
            # Download all required files
            results = manager.download_all(required_only=True, force=args.force)
            print(f"\nDownload Results:")
            for kernel_id, success in results.items():
                status = "✓ Success" if success else "✗ Failed"
                print(f"  {kernel_id}: {status}")
        else:
            # Download specified files
            results = manager.download_kernels(args.download, force=args.force)
            print(f"\nDownload Results:")
            for kernel_id, success in results.items():
                status = "✓ Success" if success else "✗ Failed"
                print(f"  {kernel_id}: {status}")
    
    elif args.mission:
        print(f"Setting up kernel files for {args.mission} mission...")
        successful = manager.setup_for_mission(args.mission, args.ephemeris)
        print(f"Successfully downloaded {len(successful)} kernel files: {', '.join(successful)}")
        
        # 显示使用的星历版本信息
        if args.ephemeris == 'de442':
            print("\nNote: Using DE442 ephemeris - includes:")
            print("  • Updated Uranus barycenter for URA182 satellite ephemeris")
            print("  • Uranus occultation data")
            print("  • Additional Mars orbiter ranging data")
            print("  • Additional Juno ranging data (4 more years)")
        elif args.ephemeris == 'de440_small':
            print("\nNote: Using DE440 small version - reduced size for storage-constrained applications")
    
    elif args.check_updates:
        updates = manager.check_updates()
        need_update = [k for k, v in updates.items() if v]
        
        if need_update:
            print(f"Kernels needing update ({len(need_update)}):")
            for kernel_id in need_update:
                kernel = manager.get_kernel(kernel_id)
                last_update = kernel.last_update.strftime('%Y-%m-%d') if kernel.last_update else "Never"
                print(f"  - {kernel_id}: {kernel.name} (Last updated: {last_update})")
            
            response = input("\nUpdate now? (y/N): ").strip().lower()
            if response == 'y':
                for kernel_id in need_update:
                    success = manager.download_kernel(kernel_id, force=True)
                    status = "Success" if success else "Failed"
                    print(f"  {kernel_id}: {status}")
        else:
            print("All kernel files are up to date")
    
    elif args.clean is not None:
        print(f"Cleaning files older than {args.clean} days...")
        if args.verbose:
            print("Files to be deleted:")
            manager.clean_cache(days_old=args.clean, dry_run=True)
        
        response = input("Confirm deletion? (y/N): ").strip().lower()
        if response == 'y':
            deleted = manager.clean_cache(days_old=args.clean, dry_run=False)
            if deleted:
                print(f"\nDeleted {len(deleted)} files:")
                for f in deleted:
                    print(f"  - {Path(f).name}")
            else:
                print("No files to clean")
        else:
            print("Clean cancelled")
    
    elif args.stats:
        stats = manager.get_stats()
        print(f"\nSPICE Kernel Statistics:")
        print(f"  Kernel Directory: {stats['kernel_dir']}")
        print(f"  Total Kernels: {stats['total_kernels']}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Total Size: {stats['total_size_mb']} MB")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
