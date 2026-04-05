"""
运行 LUNAR-SWING 桩测试的便捷入口

此模块提供运行所有桩测试的功能，验证接口设计合理性。
符合 pytest 命名规范 (test_*.py)，可被 pytest 自动发现。
"""
import sys
import os
import pytest

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

def test_stub_runner():
    """
    运行所有 lunar-swing 桩测试的入口测试函数。
    
    注意：这个测试函数本身不包含断言，它依赖于 pytest 自动发现
    并运行所有 test_*.py 文件。添加这个函数是为了：
    1. 符合 test_*.py 命名规范
    2. 提供文档说明
    3. 保持命令行接口功能
    """
    # 这个测试函数总是通过，实际测试由 pytest 自动发现机制执行
    assert True, "桩测试运行器占位符"

def run_stub_tests_cli():
    """
    Command line interface: run all stub tests
    
    Preserves original CLI functionality for manual execution
    """
    print("=" * 70)
    print("Running LUNAR-SWING Stub Tests (Stage 1: Interface Design Validation)")
    print("=" * 70)
    
    # Test file list (same as before)
    test_files = [
        'test_ephemeris_stub.py',
        'test_crtbp_stub.py', 
        'test_geopotential_stub.py',
        'test_targeter_stub.py',
        'test_stm_calculator_stub.py'
    ]
    
    # Build full paths
    test_paths = [os.path.join(os.path.dirname(__file__), f) for f in test_files]
    
    # Run tests
    args = [
        '-v',
        '--tb=short',  # Short traceback
        '--disable-warnings',
        '--capture=no'  # Show print output
    ]
    
    # Add test files
    args.extend(test_paths)
    
    print(f"\nRunning {len(test_files)} stub test files...")
    result = pytest.main(args)
    
    if result == 0:
        print("\n✅ All stub tests passed! Interface design is valid.")
    else:
        print("\n❌ Some stub tests failed. Please check interface design.")
    
    return result

if __name__ == '__main__':
    # 命令行入口
    sys.exit(run_stub_tests_cli())
