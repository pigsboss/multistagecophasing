# 05. 基础设施升级方案：接口契约与测试桩

> **本阶段目标**：定义 `lunar-swing` 项目所需的全部新基础设施模块的**接口契约**，确保它们与MCPC现有架构兼容，并为后续实现提供明确的规范。同时为每个接口创建**桩测试**，验证设计合理性。

## 5.1 升级总览

本项目将驱动MCPC框架在以下三个关键领域进行能力升级：

1. **高精度时空基准**：`HighPrecisionEphemeris` 类
2. **高保真力模型**：`UniversalCRTBP` 和 `HighOrderGeopotential` 类  
3. **专用算法工具**：`LunarSwingTargeter` 和 `STMCalculator` 类

所有新模块必须遵循MCPC的现有契约体系（特别是 `IForceModel`、`CoordinateFrame` 等）。

## 5.2 模块一：高精度星历 (`HighPrecisionEphemeris`)

### 5.2.1 设计目标
提供太阳系主要天体在指定时间、指定坐标系下的高精度位置和速度，支持 JPL DE440/DE430 等标准星历格式。

### 5.2.2 接口契约
```python
# 位置：mission_sim/core/spacetime/ephemeris/high_precision.py
class HighPrecisionEphemeris:
    """高精度星历模块，封装外部星历库（如SPICE或本地DE数据）"""
    
    def __init__(self, kernel_path: str = None):
        """
        初始化星历系统。
        
        Args:
            kernel_path: SPICE内核文件或DE数据文件路径。如果为None，使用内置简化模型。
        """
        pass
    
    def get_state(self, 
                  target_body: str, 
                  epoch: float, 
                  observer_body: str = 'earth',
                  frame: Union[str, CoordinateFrame] = 'J2000') -> np.ndarray:
        """
        获取目标天体在指定时刻、相对观测者、在指定坐标系下的状态。
        
        Args:
            target_body: 目标天体名称，如 'moon', 'sun', 'earth'
            epoch: 历元时间（秒，从J2000起算）
            observer_body: 观测者天体名称
            frame: 参考坐标系，支持字符串或CoordinateFrame枚举
            
        Returns:
            6维状态向量 [x, y, z, vx, vy, vz] (m, m/s)，在指定坐标系下
        """
        pass
    
    def get_earth_moon_rotating_state(self, epoch: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取地月旋转坐标系在J2000中的状态（用于CRTBP初始化）。
        
        Args:
            epoch: 历元时间（秒）
            
        Returns:
            (earth_state, moon_state): 地球和月球在J2000中的状态 [x,y,z,vx,vy,vz]
        """
        pass
```

### 5.2.3 桩测试示例 (`tests/lunar_swing/test_ephemeris_stub.py`)
```python
def test_high_precision_ephemeris_interface():
    """验证HighPrecisionEphemeris接口设计是否合理"""
    # 1. 测试初始化
    ephem = HighPrecisionEphemeris(kernel_path='data/de440.bsp')
    
    # 2. 测试基本状态获取（桩实现应返回固定值）
    state = ephem.get_state('moon', 0.0, 'earth', 'J2000')
    assert state.shape == (6,)
    
    # 3. 测试地月旋转系状态获取
    earth_state, moon_state = ephem.get_earth_moon_rotating_state(0.0)
    assert earth_state.shape == (6,)
    assert moon_state.shape == (6,)
    
    # 4. 测试错误处理
    with pytest.raises(ValueError):
        ephem.get_state('unknown_body', 0.0)
```

## 5.3 模块二：通用三体动力学 (`UniversalCRTBP`)

### 5.3.1 设计目标
替换现有的硬编码CRTBP实现，支持任意双主天体系统（地月、日地等），并实现为 `IForceModel` 接口。

### 5.3.2 接口契约
```python
# 位置：mission_sim/core/physics/models/threebody/universal_crtbp.py
class UniversalCRTBP(IForceModel):
    """通用圆形限制性三体问题动力学模型"""
    
    def __init__(self, 
                 primary_mass: float,
                 secondary_mass: float,
                 distance: float,
                 system_name: str = 'custom'):
        """
        初始化CRTBP系统。
        
        Args:
            primary_mass: 主天体质心 (kg)
            secondary_mass: 次天体质心 (kg) 
            distance: 双天体平均距离 (m)
            system_name: 系统标识符，如 'earth_moon', 'sun_earth'
        """
        pass
    
    @classmethod
    def earth_moon_system(cls) -> 'UniversalCRTBP':
        """创建地月系统CRTBP（便捷构造方法）"""
        pass
    
    @classmethod  
    def sun_earth_system(cls) -> 'UniversalCRTBP':
        """创建日地系统CRTBP（便捷构造方法）"""
        pass
    
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算CRTBP加速度（实现IForceModel接口）。
        
        Args:
            state: 航天器状态 [x,y,z,vx,vy,vz] (m, m/s)，在旋转坐标系中
            epoch: 当前时间（秒），用于验证相位一致性
            
        Returns:
            加速度向量 [ax, ay, az] (m/s²)
        """
        pass
    
    def jacobi_constant(self, state: np.ndarray) -> float:
        """计算雅可比常数C（用于验证能量守恒）"""
        pass
    
    def to_rotating_frame(self, state_inertial: np.ndarray, epoch: float) -> np.ndarray:
        """从惯性系转换到旋转系"""
        pass
        
    def to_inertial_frame(self, state_rotating: np.ndarray, epoch: float) -> np.ndarray:
        """从旋转系转换到惯性系"""
        pass
```

### 5.3.3 桩测试示例
```python
def test_universal_crtbp_interface():
    """验证UniversalCRTBP接口设计"""
    # 1. 测试便捷构造
    earth_moon = UniversalCRTBP.earth_moon_system()
    sun_earth = UniversalCRTBP.sun_earth_system()
    
    # 2. 测试IForceModel兼容性
    assert isinstance(earth_moon, IForceModel)
    
    # 3. 测试加速度计算接口
    test_state = np.array([1e8, 0, 0, 0, 1e3, 0])  # 示例状态
    accel = earth_moon.compute_accel(test_state, epoch=0.0)
    assert accel.shape == (3,)
    
    # 4. 测试雅可比常数计算
    C = earth_moon.jacobi_constant(test_state)
    assert isinstance(C, float)
```

## 5.4 模块三：高阶地球重力场 (`HighOrderGeopotential`)

### 5.4.1 设计目标
实现高阶地球非球形引力模型（支持EGM2008等），作为 `IForceModel` 集成到物理域。

### 5.4.2 接口契约
```python
# 位置：mission_sim/core/physics/models/gravity/high_order_geopotential.py
class HighOrderGeopotential(IForceModel):
    """高阶地球重力场模型（球谐函数展开）"""
    
    def __init__(self, 
                 degree: int = 10,
                 order: int = 10,
                 coeff_file: str = None):
        """
        初始化重力场模型。
        
        Args:
            degree: 最大阶数
            order: 最大次数（通常order <= degree）
            coeff_file: 球谐系数文件路径（如EGM2008），None则使用内置WGS84系数
        """
        pass
    
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算高阶重力场加速度（地心惯性系）。
        
        Args:
            state: 航天器状态 [x,y,z,vx,vy,vz] (m, m/s)
            epoch: 当前时间（秒）
            
        Returns:
            加速度向量 [ax, ay, az] (m/s²)
        """
        pass
    
    def set_max_degree(self, degree: int):
        """动态设置最大阶数（用于性能与精度权衡）"""
        pass
```

## 5.5 模块四：轨道设计器 (`LunarSwingTargeter`)

### 5.5.1 设计目标
提供共振摆动轨道搜索的核心算法引擎，集成打靶法、STM计算和轨道族追踪功能。

### 5.5.2 接口契约
```python
# 位置：mission_sim/core/cyber/algorithms/lunar_swing_targeter.py
class LunarSwingTargeter:
    """地月共振摆动轨道设计器"""
    
    def __init__(self, 
                 dynamics_model: Union[IForceModel, Callable],
                 integrator_type: str = 'rkf78'):
        """
        初始化轨道设计器。
        
        Args:
            dynamics_model: 动力学模型（需支持状态导数计算）
            integrator_type: 积分器类型，'rk4'、'rkf78'、'dop853'
        """
        pass
    
    def find_resonant_orbit(self,
                           resonance_ratio: Tuple[int, int],
                           initial_guess: np.ndarray,
                           target_period: float = None,
                           tol: float = 1e-8,
                           max_iter: int = 50) -> Dict:
        """
        搜索共振周期轨道。
        
        Args:
            resonance_ratio: (n,m) 共振比
            initial_guess: 6维初始状态猜测
            target_period: 目标周期（秒），None则根据共振比计算
            tol: 收敛容差
            max_iter: 最大迭代次数
            
        Returns:
            字典包含：'state'（周期轨道状态）, 'period', 'convergence_history'
        """
        pass
    
    def compute_stm(self, 
                   initial_state: np.ndarray, 
                   duration: float) -> np.ndarray:
        """
        计算状态转移矩阵。
        
        Args:
            initial_state: 初始状态
            duration: 积分时长（秒）
            
        Returns:
            6x6状态转移矩阵
        """
        pass
    
    def analyze_stability(self, orbit_state: np.ndarray, period: float) -> Dict:
        """
        分析轨道稳定性（计算单值矩阵特征值）。
        
        Returns:
            包含特征值、稳定性指标等信息的字典
        """
        pass
```

## 5.6 模块五：状态转移矩阵计算器 (`STMCalculator`)

### 5.6.1 设计目标
提供通用的STM计算工具，可被 `LunarSwingTargeter` 和其他模块复用。

### 5.6.2 接口契约
```python
# 位置：mission_sim/utils/dynamics/stm_calculator.py
class STMCalculator:
    """通用状态转移矩阵计算器"""
    
    @staticmethod
    def compute_numerical(dynamics: Callable,
                         initial_state: np.ndarray,
                         t0: float,
                         tf: float,
                         method: str = 'DOP853') -> np.ndarray:
        """
        通过数值积分计算STM。
        
        Args:
            dynamics: 状态导数函数 f(t, x) -> dx/dt
            initial_state: 初始状态
            t0, tf: 积分起止时间
            method: 积分器方法
            
        Returns:
            6x6状态转移矩阵 Φ(tf, t0)
        """
        pass
    
    @staticmethod
    def compute_analytic(dynamics_jacobian: Callable,
                        initial_state: np.ndarray,
                        t0: float,
                        tf: float) -> np.ndarray:
        """
        通过解析变分方程计算STM（要求提供雅可比函数）。
        
        Args:
            dynamics_jacobian: 雅可比函数 J(t, x) -> 6x6矩阵
            initial_state: 初始状态
            t0, tf: 积分起止时间
            
        Returns:
            6x6状态转移矩阵
        """
        pass
```

## 5.7 桩测试框架建立

在项目根目录执行以下操作：
```bash
# 1. 创建专用测试目录
mkdir -p tests/lunar_swing

# 2. 创建桩测试主文件
touch tests/lunar_swing/__init__.py
touch tests/lunar_swing/conftest.py

# 3. 为每个模块创建桩测试文件
touch tests/lunar_swing/test_ephemeris_stub.py
touch tests/lunar_swing/test_crtbp_stub.py
touch tests/lunar_swing/test_geopotential_stub.py
touch tests/lunar_swing/test_targeter_stub.py
touch tests/lunar_swing/test_stm_calculator_stub.py
```

每个桩测试文件应包含：
1. 接口验证测试（如上所示）
2. 异常处理测试
3. 与现有MCPC契约的兼容性测试

## 5.8 与现有MCPC架构的集成点

| 新模块 | 继承/实现 | 依赖现有模块 | 影响范围 |
|--------|-----------|--------------|----------|
| `HighPrecisionEphemeris` | 无 | `CoordinateFrame`, `math_tools` | 时空域 |
| `UniversalCRTBP` | `IForceModel` | `CoordinateFrame` | 物理域 |
| `HighOrderGeopotential` | `IForceModel` | 无 | 物理域 |
| `LunarSwingTargeter` | 无 | `math_tools`, `STMCalculator` | 赛博域/算法 |
| `STMCalculator` | 无 | `math_tools` | 工具层 |

## 5.9 第一阶段验收标准（基础设施设计阶段）

1. ✅ 所有5个新模块的接口契约明确定义，并通过文档评审
2. ✅ 为每个接口创建了桩测试，测试通过（验证接口设计合理性）
3. ✅ 确认新模块与MCPC现有契约体系兼容
4. ✅ 建立 `tests/lunar_swing/` 目录结构，为后续开发做好准备

## 5.10 下一步：从设计到实现

完成本阶段后，第二阶段（Sprint 3-4）将：
1. 基于这些接口契约，开始实现 `UniversalCRTBP` 和 `LunarSwingTargeter`
2. 使用桩测试作为起点，逐步替换为真实实现
3. 每完成一个功能点，立即添加单元测试，确保测试始终通过

---
*基础设施设计完成。下一步：开始第二阶段的核心算法实现。*
