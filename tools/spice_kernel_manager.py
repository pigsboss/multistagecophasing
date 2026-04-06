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
7. **NEW**: Smart location - automatically finds kernel files even if NAIF website paths change

Proxy Usage Examples:
  1. Command line: --proxy http://proxy.example.com:8080
  2. Environment variable: export HTTPS_PROXY=https://proxy.example.com:8080
  3. With authentication: --proxy http://username:password@proxy.example.com:8080

Common Issues & Solutions:
  1. Connection timeout: Increase timeout with --timeout 300
  2. Proxy required: Use --proxy option or set HTTPS_PROXY environment variable
  3. Slow downloads: Use --verbose to monitor progress, consider using a faster proxy
  4. Path changes: Smart location automatically finds new paths on NAIF website

Author: MCPC Development Team
Version: 1.2.0 (Added smart location)
"""

import os
import sys
import time
import hashlib
import shutil
import json
import gzip
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

try:
    import requests
    from tqdm import tqdm
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Warning: Missing requests or tqdm library, automatic download unavailable")
    print("Please install: pip install requests tqdm")

# 设置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
    search_patterns: List[str] = field(default_factory=list)  # 搜索模式（正则表达式）
    search_paths: List[str] = field(default_factory=list)     # 搜索路径


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
    proxy: Optional[str] = None    # HTTP/HTTPS代理 (例如: https://proxy.example.com:8080)
    proxy_auth: Optional[Tuple[str, str]] = None  # 代理认证 (用户名, 密码)
    enable_smart_location: bool = True  # 启用智能定位
    smart_location_depth: int = 3       # 智能定位搜索深度


class NAIFKernelLocator:
    """NAIF Kernel 智能定位器 - 自动处理路径变化"""
    
    def __init__(self, base_url: str = "https://naif.jpl.nasa.gov/pub/naif/",
                 proxy: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/') + '/'
        self.timeout = timeout
        
        # 创建会话，优先使用 HTTPS 代理
        self.session = requests.Session()
        
        # 配置代理 - 优先使用 HTTPS_PROXY
        if proxy:
            # 确保代理 URL 格式正确
            if not proxy.startswith(('http://', 'https://')):
                proxy = f"https://{proxy}"  # 默认使用 HTTPS
            self.session.proxies = {'http': proxy, 'https': proxy}
            logger.debug(f"Locator using proxy: {proxy}")
        
        self.session.headers.update({
            'User-Agent': 'MCPC-SPICE-Smart-Locator/1.2.0 (+https://github.com/mcpc-project)'
        })
        
        # 缓存已访问页面
        self._visited = set()
        self._cache = {}
    
    def find_kernel(self, search_patterns: List[str], search_paths: List[str],
                   prefer_version: Optional[str] = None, max_depth: int = 3) -> Optional[str]:
        """
        智能查找 kernel 文件
        
        Args:
            search_patterns: 文件名模式列表（正则表达式）
            search_paths: 搜索路径列表
            prefer_version: 首选版本号
            max_depth: 最大搜索深度
            
        Returns:
            找到的文件 URL，或 None
        """
        if not HAS_DEPENDENCIES:
            logger.warning("Missing requests library, smart location unavailable")
            return None
        
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in search_patterns]
        
        for path in search_paths:
            start_url = urljoin(self.base_url, path)
            if not start_url.endswith('/'):
                start_url += '/'
            
            logger.info(f"Searching for kernel in: {start_url}")
            
            found = self._search_recursive(start_url, compiled_patterns, 
                                         prefer_version, max_depth, current_depth=0)
            if found:
                logger.info(f"Found kernel at: {found}")
                return found
        
        return None
    
    def _search_recursive(self, url: str, patterns: List[re.Pattern],
                         prefer_version: Optional[str], max_depth: int,
                         current_depth: int) -> Optional[str]:
        """递归搜索目录"""
        if current_depth > max_depth or url in self._visited:
            return None
        
        self._visited.add(url)
        
        try:
            logger.debug(f"Searching URL (depth {current_depth}): {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # 解析目录页面
            items = self._parse_apache_directory(response.text, url)
            
            # 先检查文件（优先于目录）
            files = [item for item in items if not item['is_dir']]
            dirs = [item for item in items if item['is_dir']]
            
            # 检查文件
            for item in files:
                for pattern in patterns:
                    if pattern.search(item['name']):
                        # 版本检查
                        if self._is_preferred_version(item['name'], prefer_version):
                            logger.debug(f"Found matching file: {item['name']}")
                            return item['url']
            
            # 递归搜索子目录
            for item in dirs:
                found = self._search_recursive(
                    item['url'], patterns, prefer_version, 
                    max_depth, current_depth + 1
                )
                if found:
                    return found
                    
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to access {url}: {e}")
        except Exception as e:
            logger.debug(f"Error searching {url}: {e}")
        
        return None
    
    def _parse_apache_directory(self, html: str, base_url: str) -> List[Dict]:
        """解析 Apache 目录列表"""
        items = []
        
        # Apache 目录列表常见模式
        patterns = [
            # 标准格式: <a href="filename">filename</a> date size
            r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>\s+(\d{2,4}-\w{3}-\d{4}\s+\d{2}:\d{2})\s+([\d\.\-]+[KMG]?)',
            # 简单链接格式
            r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, html, re.IGNORECASE):
                href = match.group(1)
                text = match.group(2) if len(match.groups()) > 1 else href
                
                # 跳过特殊链接
                if href in ['../', '.', '..']:
                    continue
                
                # 确定类型
                is_dir = href.endswith('/')
                if is_dir:
                    href = href.rstrip('/')
                    text = text.rstrip('/')
                
                # 构建完整 URL
                full_url = urljoin(base_url + ('/' if not base_url.endswith('/') else ''), href)
                
                items.append({
                    'name': text,
                    'url': full_url,
                    'is_dir': is_dir
                })
        
        return items
    
    def _is_preferred_version(self, filename: str, prefer_version: Optional[str]) -> bool:
        """检查是否为首选版本"""
        if not prefer_version:
            return True
        
        # 提取版本号
        version_match = re.search(r'\d{3,4}', filename)
        if not version_match:
            return True  # 没有版本号，视为匹配
        
        file_version = version_match.group()
        
        try:
            file_ver_num = int(file_version)
            pref_ver_num = int(prefer_version)
            
            # 允许相同或更新版本
            return file_ver_num >= pref_ver_num
        except ValueError:
            return True


class SPICEKernelManager:
    """
    SPICE核文件管理器
    
    自动化管理SPICE核文件的下载、更新和验证。
    新增智能定位功能：自动处理NAIF网站路径变化。
    """
    
    # 常用核文件定义 - 增强版本，包含搜索模式
    COMMON_KERNELS = {
        'de440': {
            'name': 'de440.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp',
            'description': 'DE440 Planetary Ephemeris (1550-2650) - Well-established standard',
            'required': True,
            'size': 135_000_000,
            'compressed': False,
            'search_patterns': [r'de44[0-9]\.bsp', r'de44[0-9][a-z]?\.bsp'],
            'search_paths': ['generic_kernels/spk/planets/', 'generic_kernels/spk/']
        },
        'de442': {
            'name': 'de442.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442.bsp',
            'description': 'DE442 Planetary Ephemeris (1549-2650) - Updated Uranus barycenter with occultation data and extended Mars/Juno ranging',
            'required': False,
            'size': 150_000_000,
            'compressed': False,
            'search_patterns': [r'de44[0-9]\.bsp'],
            'search_paths': ['generic_kernels/spk/planets/', 'generic_kernels/spk/']
        },
        'de440_small': {
            'name': 'de440s.bsp',
            'category': 'ephemeris',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp',
            'description': 'DE440 Simplified Planetary Ephemeris (1550-2650) - For storage-constrained applications',
            'required': False,
            'size': 7_000_000,
            'compressed': False,
            'search_patterns': [r'de44[0-9]s\.bsp', r'de\d{3}s\.bsp'],
            'search_paths': ['generic_kernels/spk/planets/', 'generic_kernels/spk/']
        },
        'pck00010': {
            'name': 'pck00010.tpc',
            'category': 'frame',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc',
            'description': '行星常数和方向信息',
            'required': True,
            'size': 2_000_000,
            'compressed': False,
            'search_patterns': [r'pck\d{5}\.tpc', r'pck\d{5}_[a-z0-9]+\.tpc'],
            'search_paths': ['generic_kernels/pck/', 'generic_kernels/pck/old_versions/']
        },
        'naif0012': {
            'name': 'naif0012.tls',
            'category': 'clock',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls',
            'description': '时间系统转换',
            'required': True,
            'size': 100_000,
            'compressed': False,
            'search_patterns': [r'naif\d{4}\.tls', r'latest_leapseconds\.tls'],
            'search_paths': ['generic_kernels/lsk/', 'generic_kernels/lsk/old_versions/']
        },
        'earth_200101': {
            'name': 'earth_200101_230317_230317.bpc',
            'category': 'attitude',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_200101_230317_230317.bpc',
            'description': '地球姿态模型（2001-2023）',
            'required': False,
            'size': 5_000_000,
            'compressed': False,
            'search_patterns': [r'earth_[0-9_]+\.bpc', r'earth_[0-9_]+_predict\.bpc'],
            'search_paths': ['generic_kernels/pck/', 'generic_kernels/pck/old_versions/']
        },
        'earth_070425': {
            'name': 'earth_070425_370426_predict.bpc',
            'category': 'attitude',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_070425_370426_predict.bpc',
            'description': '地球姿态预测模型',
            'required': False,
            'size': 3_000_000,
            'compressed': False,
            'search_patterns': [r'earth_[0-9_]+\.bpc', r'earth_[0-9_]+_predict\.bpc'],
            'search_paths': ['generic_kernels/pck/', 'generic_kernels/pck/old_versions/']
        },
        'moon_pa': {
            'name': 'moon_pa_de440_200625.bpc',
            'category': 'attitude',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc',
            'description': '月球姿态模型',
            'required': False,
            'size': 2_000_000,
            'compressed': False,
            'search_patterns': [r'moon_[a-z0-9_]+\.bpc', r'moon_pa_[a-z0-9_]+\.bpc'],
            'search_paths': ['generic_kernels/pck/', 'generic_kernels/pck/old_versions/']
        },
        'latest_leapseconds': {
            'name': 'latest_leapseconds.tls',
            'category': 'clock',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls',
            'description': '最新闰秒表',
            'required': False,
            'size': 50_000,
            'compressed': False,
            'search_patterns': [r'latest_leapseconds\.tls', r'leapseconds\.tls'],
            'search_paths': ['generic_kernels/lsk/', 'generic_kernels/lsk/old_versions/']
        },
        'frames': {
            'name': 'frames.tf',
            'category': 'frame',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/frames/frames.tf',
            'description': '坐标系定义',
            'required': True,
            'size': 500_000,
            'compressed': False,
            'search_patterns': [r'frames\.tf', r'frames_\d+\.tf'],
            'search_paths': ['generic_kernels/fk/frames/', 'generic_kernels/fk/']
        },
        'earth_frames': {
            'name': 'earth_000101.tf',
            'category': 'frame',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/planets/earth_000101.tf',
            'description': '地球坐标系定义',
            'required': False,
            'size': 200_000,
            'compressed': False,
            'search_patterns': [r'earth_[a-z0-9_]+\.tf', r'iau_earth\.tf'],
            'search_paths': ['generic_kernels/fk/planets/', 'generic_kernels/fk/']
        },
        'moon_frames': {
            'name': 'moon_080317.tf',
            'category': 'frame',
            'url': 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_080317.tf',
            'description': '月球坐标系定义',
            'required': False,
            'size': 150_000,
            'compressed': False,
            'search_patterns': [r'moon_[a-z0-9_]+\.tf', r'iau_moon\.tf'],
            'search_paths': ['generic_kernels/fk/satellites/', 'generic_kernels/fk/']
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
        
        # 检查环境变量中的代理设置 - 优先使用 HTTPS_PROXY
        if not self.config.proxy:
            # 检查常见的环境变量，优先使用 HTTPS_PROXY
            for env_var in ['HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy']:
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
        
        # Initialize smart locator
        self.locator = None
        if self.config.enable_smart_location and HAS_DEPENDENCIES:
            self.locator = NAIFKernelLocator(
                proxy=self.config.proxy,
                timeout=self.config.timeout
            )
        
        if not HAS_DEPENDENCIES:
            if self.config.auto_download:
                warnings.warn("Missing requests or tqdm library, automatic download unavailable")
        
        if self.verbose:
            print(f"[SPICEKernelManager] Initialization complete. Kernel directory: {self.kernel_dir}")
            print(f"[SPICEKernelManager] Available kernels: {len(self._kernels)}")
            print(f"[SPICEKernelManager] Smart location: {'Enabled' if self.config.enable_smart_location else 'Disabled'}")
    
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
                compressed=kernel_data.get('compressed', False),
                search_patterns=kernel_data.get('search_patterns', []),
                search_paths=kernel_data.get('search_paths', [])
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
                'description': kernel.description,
                'url': kernel.url  # 保存当前 URL
            }
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.verbose:
                print(f"[SPICEKernelManager] Failed to save state: {e}")
    
    # =========================== 智能定位功能 ===========================
    
    def smart_download_kernel(self, kernel_id: str, force: bool = False) -> Tuple[bool, Optional[str]]:
        """
        智能下载 kernel 文件 - 自动处理路径变化
        
        Args:
            kernel_id: Kernel identifier
            force: Force re-download
            
        Returns:
            Tuple[bool, Optional[str]]: (success, actual_url_or_error_message)
        """
        if kernel_id not in self._kernels:
            return False, f"Kernel '{kernel_id}' not found"
        
        kernel = self._kernels[kernel_id]
        
        # 检查是否已存在且不需要强制重新下载
        if kernel.local_path.exists() and not force:
            if self.verbose:
                print(f"[SPICEKernelManager] File already exists: {kernel.local_path}")
            return True, kernel.url
        
        # 1. 首先尝试直接下载（使用已知 URL）
        if self.verbose:
            print(f"[SPICEKernelManager] Trying direct download: {kernel.name}")
        
        direct_success = self._download_with_retry(kernel.url, kernel.local_path, kernel.name)
        
        if direct_success:
            if self.verbose:
                print(f"[SPICEKernelManager] Direct download successful: {kernel.name}")
            kernel.last_update = datetime.now()
            self._save_state()
            return True, kernel.url
        
        # 2. 如果直接下载失败且启用了智能定位，尝试智能定位
        if self.config.enable_smart_location and self.locator:
            if self.verbose:
                print(f"[SPICEKernelManager] Direct download failed, trying smart location for {kernel.name}")
            
            # 获取搜索模式和路径
            search_patterns = kernel.search_patterns
            search_paths = kernel.search_paths
            
            # 如果没有定义搜索模式，使用默认模式
            if not search_patterns:
                search_patterns = self._get_default_search_patterns(kernel_id)
            
            if not search_paths:
                search_paths = self._get_default_search_paths(kernel.category)
            
            if self.verbose:
                print(f"[SPICEKernelManager] Searching with patterns: {search_patterns}")
                print(f"[SPICEKernelManager] Searching in paths: {search_paths}")
            
            # 智能定位
            found_url = self.locator.find_kernel(
                search_patterns=search_patterns,
                search_paths=search_paths,
                prefer_version=self._get_preferred_version(kernel_id),
                max_depth=self.config.smart_location_depth
            )
            
            if found_url:
                if self.verbose:
                    print(f"[SPICEKernelManager] Found new location: {found_url}")
                
                # 更新 kernel URL
                old_url = kernel.url
                kernel.url = found_url
                
                # 尝试从新位置下载
                smart_success = self._download_with_retry(found_url, kernel.local_path, kernel.name)
                
                if smart_success:
                    if self.verbose:
                        print(f"[SPICEKernelManager] Smart download successful: {kernel.name}")
                    kernel.last_update = datetime.now()
                    self._save_state()
                    return True, found_url
                else:
                    # 恢复原始 URL
                    kernel.url = old_url
                    if self.verbose:
                        print(f"[SPICEKernelManager] Smart download failed, restored original URL")
        
        # 3. 所有尝试都失败
        error_msg = f"Failed to download {kernel.name}. "
        error_msg += "Possible reasons:\n"
        error_msg += "  1. Network connectivity issues\n"
        error_msg += "  2. NAIF website may be down or restructured\n"
        error_msg += "  3. Proxy configuration may be incorrect\n"
        error_msg += "  4. File may have been moved or renamed\n\n"
        error_msg += "Troubleshooting:\n"
        error_msg += "  1. Check internet connection\n"
        error_msg += "  2. Verify proxy settings (use --proxy option)\n"
        error_msg += "  3. Try with --verbose flag for more details\n"
        error_msg += "  4. Manually download from: https://naif.jpl.nasa.gov/naif/data.html"
        
        return False, error_msg
    
    def _download_with_retry(self, url: str, dest_path: Path, kernel_name: str) -> bool:
        """带重试的下载函数"""
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                if self.verbose:
                    print(f"[SPICEKernelManager] Retry {attempt}/{self.config.max_retries}...")
                time.sleep(2 ** attempt)  # 指数退避
            
            success = self._download_file(url, dest_path, kernel_name)
            if success:
                return True
        
        return False
    
    def _get_default_search_patterns(self, kernel_id: str) -> List[str]:
        """获取默认搜索模式"""
        patterns_map = {
            'de440': [r'de44[0-9]\.bsp', r'de44[0-9][a-z]?\.bsp'],
            'de442': [r'de44[0-9]\.bsp'],
            'de440_small': [r'de44[0-9]s\.bsp', r'de\d{3}s\.bsp'],
            'pck00010': [r'pck\d{5}\.tpc', r'pck\d{5}_[a-z0-9]+\.tpc'],
            'naif0012': [r'naif\d{4}\.tls', r'latest_leapseconds\.tls'],
            'latest_leapseconds': [r'latest_leapseconds\.tls', r'leapseconds\.tls'],
            'earth_200101': [r'earth_[0-9_]+\.bpc'],
            'earth_070425': [r'earth_[0-9_]+\.bpc'],
            'moon_pa': [r'moon_[a-z0-9_]+\.bpc'],
            'frames': [r'frames\.tf', r'frames_\d+\.tf'],
            'earth_frames': [r'earth_[a-z0-9_]+\.tf', r'iau_earth\.tf'],
            'moon_frames': [r'moon_[a-z0-9_]+\.tf', r'iau_moon\.tf'],
        }
        
        return patterns_map.get(kernel_id, [r'.*'])
    
    def _get_default_search_paths(self, category: str) -> List[str]:
        """获取默认搜索路径"""
        base_paths = {
            'ephemeris': ['generic_kernels/spk/planets/', 'generic_kernels/spk/'],
            'clock': ['generic_kernels/lsk/', 'generic_kernels/lsk/old_versions/'],
            'frame': ['generic_kernels/pck/', 'generic_kernels/pck/old_versions/'],
            'attitude': ['generic_kernels/pck/', 'generic_kernels/pck/old_versions/'],
            'fk': ['generic_kernels/fk/frames/', 'generic_kernels/fk/planets/', 
                  'generic_kernels/fk/satellites/', 'generic_kernels/fk/']
        }
        
        return base_paths.get(category, ['generic_kernels/'])
    
    def _get_preferred_version(self, kernel_id: str) -> Optional[str]:
        """获取首选版本号"""
        versions = {
            'de440': '440',
            'de442': '442',
            'de440_small': '440',
            'pck00010': '00010',
            'naif0012': '0012',
            'earth_200101': '200101',
            'earth_070425': '070425',
            'moon_pa': 'de440',
            'frames': '',
            'earth_frames': '000101',
            'moon_frames': '080317',
        }
        return versions.get(kernel_id)
    
    def smart_download_kernels(self, kernel_ids: List[str], force: bool = False) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        批量智能下载 kernel 文件
        
        Args:
            kernel_ids: List of kernel IDs
            force: Force re-download
            
        Returns:
            Dict[str, Tuple[bool, Optional[str]]]: 下载结果 {kernel_id: (success, url_or_error)}
        """
        results = {}
        
        if self.verbose:
            print(f"[SPICEKernelManager] Starting smart batch download of {len(kernel_ids)} files")
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_downloads) as executor:
            future_to_kernel = {
                executor.submit(self.smart_download_kernel, kernel_id, force): kernel_id
                for kernel_id in kernel_ids
            }
            
            for future in as_completed(future_to_kernel):
                kernel_id = future_to_kernel[future]
                try:
                    results[kernel_id] = future.result()
                except Exception as e:
                    print(f"[SPICEKernelManager] Error downloading {kernel_id}: {e}")
                    results[kernel_id] = (False, str(e))
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r[0])
        
        if self.verbose:
            print(f"[SPICEKernelManager] Smart batch download complete: {success_count}/{len(kernel_ids)} succeeded")
        
        return results
    
    def smart_download_all(self, required_only: bool = True, force: bool = False) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        智能下载所有核文件
        
        Args:
            required_only: 仅下载必需的核文件
            force: 强制重新下载
            
        Returns:
            Dict[str, Tuple[bool, Optional[str]]]: 下载结果 {核ID: (是否成功, URL或错误信息)}
        """
        kernel_ids = []
        for kernel_id, kernel in self._kernels.items():
            if required_only and not kernel.required:
                continue
            kernel_ids.append(kernel_id)
        
        return self.smart_download_kernels(kernel_ids, force=force)
    
    # =========================== 原有功能（保持兼容） ===========================
    
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
                    'description': kernel.description,
                    'smart_location': bool(kernel.search_patterns)
                })
            
            return kernels_info
    
    def get_kernel(self, kernel_id: str) -> Optional[KernelInfo]:
        """获取特定核文件信息"""
        with self._lock:
            return self._kernels.get(kernel_id)
    
    # 保留原有 download_kernel 方法以保持兼容
    def download_kernel(self, kernel_id: str, force: bool = False) -> bool:
        """
        下载单个 kernel 文件（兼容原有接口）
        
        Args:
            kernel_id: Kernel identifier
            force: Force re-download
            
        Returns:
            bool: Whether download succeeded
        """
        success, _ = self.smart_download_kernel(kernel_id, force)
        return success
    
    # 保留原有 download_kernels 方法以保持兼容
    def download_kernels(self, kernel_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """
        批量下载 kernel 文件（兼容原有接口）
        
        Args:
            kernel_ids: List of kernel IDs
            force: Force re-download
            
        Returns:
            Dict[str, bool]: Download results {kernel_id: success}
        """
        results = self.smart_download_kernels(kernel_ids, force)
        # 转换为原有格式
        return {kernel_id: success for kernel_id, (success, _) in results.items()}
    
    # 保留原有 download_all 方法以保持兼容
    def download_all(self, required_only: bool = True, force: bool = False) -> Dict[str, bool]:
        """
        下载所有核文件（兼容原有接口）
        
        Args:
            required_only: 仅下载必需的核文件
            force: 强制重新下载
            
        Returns:
            Dict[str, bool]: 下载结果 {核ID: 是否成功}
        """
        results = self.smart_download_all(required_only, force)
        # 转换为原有格式
        return {kernel_id: success for kernel_id, (success, _) in results.items()}
    
    def _download_file(self, url: str, dest_path: Path, kernel_name: str) -> bool:
        """下载单个文件（内部方法）"""
        try:
            # Create temporary file
            temp_file = dest_path.with_suffix('.downloading')
            
            # 配置请求参数
            headers = {'User-Agent': 'MCPC-SPICE-Kernel-Manager/1.2.0'}
            
            # 配置代理 - 优先使用 HTTPS
            proxies = None
            if self.config.proxy:
                # 确保代理 URL 格式正确
                proxy_url = self.config.proxy
                if not proxy_url.startswith(('http://', 'https://')):
                    proxy_url = f"https://{proxy_url}"  # 默认使用 HTTPS
                
                proxies = {
                    'http': proxy_url,
                    'https': proxy_url
                }
                if self.verbose:
                    print(f"[SPICEKernelManager] Using proxy: {proxy_url}")
        
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
    
    # =========================== 其他原有方法保持不变 ===========================
    # 以下是原有代码的其余部分，为了简洁，我只保留方法签名和关键修改
    # 完整代码需要包含原有的所有方法
    
    def _decompress_file(self, compressed_path: Path, dest_path: Path) -> bool:
        """解压文件"""
        # ... 原有代码不变 ...
        try:
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            print(f"[SPICEKernelManager] Decompression failed {compressed_path}: {e}")
            return False
    
    def check_updates(self) -> Dict[str, bool]:
        """检查核文件更新"""
        # ... 原有代码不变 ...
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
        """验证 kernel 文件完整性"""
        # ... 原有代码不变 ...
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
        """设置特定任务的内核文件"""
        # ... 原有代码不变 ...
        # 验证星历版本选择
        valid_ephemeris = ['de440', 'de442', 'de440_small']
        if ephemeris not in valid_ephemeris:
            print(f"[SPICEKernelManager] Warning: Unknown ephemeris version '{ephemeris}', using 'de440'")
            ephemeris = 'de440'
        
        # 根据任务类型和星历版本选择内核
        mission_kernels = {
            'earth_moon': [ephemeris, 'pck00010', 'naif0012', 'frames', 'earth_frames', 'moon_frames'],
            'deep_space': [ephemeris, 'pck00010', 'naif0012', 'frames'],
            'mars': [ephemeris, 'pck00010', 'naif0012', 'frames'],
            'custom': [ephemeris, 'pck00010', 'naif0012', 'frames'],
            'lightweight': ['de440_small', 'pck00010', 'naif0012', 'frames']
        }
        
        if mission_type not in mission_kernels:
            print(f"[SPICEKernelManager] Warning: Unknown mission type '{mission_type}', using 'custom'")
            mission_type = 'custom'
        
        kernel_ids = mission_kernels[mission_type]
        
        if self.verbose:
            print(f"[SPICEKernelManager] Setting up kernel files for {mission_type} mission")
            print(f"[SPICEKernelManager] Using ephemeris: {ephemeris}")
        
        # 使用智能下载
        results = self.smart_download_kernels(kernel_ids)
        
        successful = [kernel_id for kernel_id, (success, _) in results.items() if success]
        
        return successful
    
    def get_kernel_paths(self, mission_type: str = "earth_moon", 
                        ephemeris: str = "de440") -> List[str]:
        """获取特定任务的内核文件路径"""
        # ... 原有代码不变 ...
        # 验证星历版本选择
        valid_ephemeris = ['de440', 'de442', 'de440_small']
        if ephemeris not in valid_ephemeris:
            print(f"[SPICEKernelManager] Warning: Unknown ephemeris version '{ephemeris}', using 'de440'")
            ephemeris = 'de440'
        
        # 根据任务类型和星历版本选择内核
        mission_kernels = {
            'earth_moon': [ephemeris, 'pck00010', 'naif0012', 'frames', 'earth_frames', 'moon_frames'],
            'deep_space': [ephemeris, 'pck00010', 'naif0012', 'frames'],
            'mars': [ephemeris, 'pck00010', 'naif0012', 'frames'],
            'custom': [ephemeris, 'pck00010', 'naif0012', 'frames'],
            'lightweight': ['de440_small', 'pck00010', 'naif0012', 'frames']
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
    
    def test_connection(self, url: str = "https://naif.jpl.nasa.gov/pub/naif/", timeout: int = 10) -> bool:
        """测试网络连接"""
        try:
            if self.verbose:
                print(f"[SPICEKernelManager] Testing connection to {url}...")
            
            headers = {'User-Agent': 'MCPC-SPICE-Kernel-Manager/1.2.0'}
            proxies = None
            if self.config.proxy:
                # 确保代理 URL 格式正确
                proxy_url = self.config.proxy
                if not proxy_url.startswith(('http://', 'https://')):
                    proxy_url = f"https://{proxy_url}"
                
                proxies = {
                    'http': proxy_url,
                    'https': proxy_url
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
                f"size={stats['total_size_mb']} MB, directory={self.kernel_dir}, "
                f"smart_location={'on' if self.config.enable_smart_location else 'off'})")


# 命令行接口 - 添加新选项
def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SPICE Kernel Manager with Smart Location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --download
  %(prog)s --mission earth_moon
  %(prog)s --check-updates
  %(prog)s --clean 30
  %(prog)s --proxy https://proxy.example.com:8080
  %(prog)s --smart-download            # 使用智能定位下载
  %(prog)s --no-smart-location         # 禁用智能定位
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all kernel files")
    parser.add_argument("--download", nargs="*", metavar="KERNEL_ID", 
                       help="Download specified kernel files (download all required if not specified)")
    parser.add_argument("--smart-download", nargs="*", metavar="KERNEL_ID",
                       help="Smart download with automatic location (use smart location)")
    parser.add_argument("--mission", choices=["earth_moon", "deep_space", "mars", "custom", "lightweight"], 
                       help="Set up kernel files for specific mission")
    parser.add_argument("--ephemeris", choices=["de440", 'de442', 'de440_small'], default="de440",
                       help="Ephemeris version (default: de440)")
    parser.add_argument("--check-updates", action="store_true", help="Check for updates")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--clean", type=int, nargs="?", metavar="DAYS", const=30, 
                       help="Clean files older than specified days (default: 30)")
    parser.add_argument("--dir", default="~/.mission_sim/spice_kernels", 
                       help="Kernel directory (default: ~/.mission_sim/spice_kernels)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--proxy", help="HTTP/HTTPS proxy (e.g., https://proxy.example.com:8080)")
    parser.add_argument("--timeout", type=int, default=30, 
                       help="Download timeout in seconds (default: 30)")
    parser.add_argument("--no-smart-location", action="store_true",
                       help="Disable smart location feature")
    parser.add_argument("--smart-depth", type=int, default=3,
                       help="Smart location search depth (default: 3)")
    
    args = parser.parse_args()
    
    # Create manager
    config = KernelConfig(
        kernel_dir=Path(args.dir).expanduser(),
        verbose=args.verbose,
        proxy=args.proxy,
        timeout=args.timeout,
        enable_smart_location=not args.no_smart_location,
        smart_location_depth=args.smart_depth
    )
    
    manager = SPICEKernelManager(config)
    
    # Execute commands
    if args.list:
        kernels = manager.list_kernels()
        print(f"\nSPICE Kernel File List ({len(kernels)}):")
        print("=" * 120)
        print(f"{'ID':20} {'Name':25} {'Category':12} {'Size':>8} {'Smart':6} {'Status':6} {'Last Update':20} {'Description'}")
        print("-" * 120)
        for k in kernels:
            status = "✓" if k['exists'] else "✗"
            smart = "✓" if k['smart_location'] else "✗"
            size_display = k['size_mb'] if k['size_mb'] != "unknown" else "  N/A  "
            print(f"{k['id']:20} {k['name']:25} {k['category']:12} {size_display:>8} MB "
                  f"{smart:6} {status:6} {k['last_update']:20} {k['description']}")
        print("=" * 120)
        print(f"Smart location: {'Enabled' if config.enable_smart_location else 'Disabled'}")
    
    elif args.smart_download is not None:
        # 智能下载
        if len(args.smart_download) == 0:
            # 下载所有必需文件
            results = manager.smart_download_all(required_only=True, force=args.force)
            print(f"\nSmart Download Results:")
            for kernel_id, (success, url_or_error) in results.items():
                status = "✓ Success" if success else "✗ Failed"
                url_display = f" ({url_or_error})" if success and url_or_error else ""
                print(f"  {kernel_id}: {status}{url_display}")
                if not success and url_or_error and "Troubleshooting" in url_or_error:
                    print(f"    Details: {url_or_error.split('Troubleshooting')[0]}")
        else:
            # 下载指定文件
            results = manager.smart_download_kernels(args.smart_download, force=args.force)
            print(f"\nSmart Download Results:")
            for kernel_id, (success, url_or_error) in results.items():
                status = "✓ Success" if success else "✗ Failed"
                url_display = f" ({url_or_error})" if success and url_or_error else ""
                print(f"  {kernel_id}: {status}{url_display}")
    
    elif args.download is not None:
        # 传统下载（保持兼容）
        if len(args.download) == 0:
            results = manager.download_all(required_only=True, force=args.force)
            print(f"\nDownload Results (traditional):")
            for kernel_id, success in results.items():
                status = "✓ Success" if success else "✗ Failed"
                print(f"  {kernel_id}: {status}")
        else:
            results = manager.download_kernels(args.download, force=args.force)
            print(f"\nDownload Results (traditional):")
            for kernel_id, success in results.items():
                status = "✓ Success" if success else "✗ Failed"
                print(f"  {kernel_id}: {status}")
    
    elif args.mission:
        print(f"Setting up kernel files for {args.mission} mission...")
        successful = manager.setup_for_mission(args.mission, args.ephemeris)
        print(f"Successfully downloaded {len(successful)} kernel files: {', '.join(successful)}")
    
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
                print("Using smart download for updates...")
                results = manager.smart_download_kernels(need_update, force=True)
                for kernel_id, (success, url) in results.items():
                    status = "Success" if success else "Failed"
                    url_display = f" from {url}" if success and url else ""
                    print(f"  {kernel_id}: {status}{url_display}")
        else:
            print("All kernel files are up to date")
    
    elif args.stats:
        stats = manager.get_stats()
        print(f"\nSPICE Kernel Statistics:")
        print(f"  Kernel Directory: {stats['kernel_dir']}")
        print(f"  Total Kernels: {stats['total_kernels']}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Total Size: {stats['total_size_mb']} MB")
        print(f"  Smart Location: {'Enabled' if config.enable_smart_location else 'Disabled'}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
