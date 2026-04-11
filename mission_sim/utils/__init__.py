"""
MCPC 通用工具库
包含数学、日志、动态系统等通用工具。
"""

from .math_tools import (
    normalize_vector,
    get_lqr_gain,
    elements_to_cartesian,
    inertial_to_rotating,
    rotating_to_inertial,
    inertial_to_earth_moon_rotating,
    earth_moon_rotating_to_inertial,
    get_earth_moon_system_parameters,
    compute_lvlh_dcm,
    absolute_to_lvlh,
    lvlh_to_absolute,
    # 新增的通用数学工具
    solve_kepler_equation_batch,
    solve_kepler_equation_scalar,
    orbital_elements_to_cartesian_batch
)

__all__ = [
    # 数学工具
    'normalize_vector',
    'get_lqr_gain',
    'elements_to_cartesian',
    'inertial_to_rotating',
    'rotating_to_inertial',
    'inertial_to_earth_moon_rotating',
    'earth_moon_rotating_to_inertial',
    'get_earth_moon_system_parameters',
    'compute_lvlh_dcm',
    'absolute_to_lvlh',
    'lvlh_to_absolute',
    # 新增的通用数学工具
    'solve_kepler_equation_batch',
    'solve_kepler_equation_scalar',
    'orbital_elements_to_cartesian_batch',
]
