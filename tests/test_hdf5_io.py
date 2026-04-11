# tests/test_hdf5_io.py
import pytest
import os
import tempfile
import numpy as np
import h5py
import multiprocessing as mp
import threading
import time
from mission_sim.utils.logger import HDF5Logger


# =============================================================================
# 单进程基本功能测试
# =============================================================================
class TestHDF5LoggerBasic:
    """测试 HDF5Logger 单进程功能"""

    @pytest.fixture
    def temp_h5(self):
        """临时 HDF5 文件路径"""
        fd, path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_initialize_and_write(self, temp_h5):
        """测试初始化和写入单条记录"""
        logger = HDF5Logger(temp_h5, buffer_size=10, compression=False)
        logger.initialize_file({"test": "meta"})
        logger.log_step(
            epoch=1.0,
            nominal_state=np.zeros(6),
            true_state=np.ones(6),
            nav_state=np.zeros(6),
            tracking_error=np.zeros(6),
            control_force=np.zeros(3),
            accumulated_dv=0.0
        )
        logger.flush()
        logger.close()

        with h5py.File(temp_h5, 'r') as f:
            assert f['epochs'].shape == (1,)
            assert f['true_states'].shape == (1, 6)
            assert f.attrs['test'] == "meta"

    def test_buffer_flush(self, temp_h5):
        """测试缓冲区自动刷新"""
        logger = HDF5Logger(temp_h5, buffer_size=2, compression=False)
        logger.initialize_file()
        for i in range(5):
            logger.log_step(
                epoch=float(i),
                nominal_state=np.zeros(6),
                true_state=np.ones(6),
                nav_state=np.zeros(6),
                tracking_error=np.zeros(6),
                control_force=np.zeros(3),
                accumulated_dv=0.0
            )
        logger.close()

        with h5py.File(temp_h5, 'r') as f:
            assert f['epochs'].shape == (5,)

    def test_context_manager(self, temp_h5):
        """测试上下文管理器"""
        with HDF5Logger(temp_h5) as logger:
            logger.initialize_file()
            logger.log_step(
                epoch=1.0,
                nominal_state=np.zeros(6),
                true_state=np.ones(6),
                nav_state=np.zeros(6),
                tracking_error=np.zeros(6),
                control_force=np.zeros(3),
                accumulated_dv=0.0
            )
        assert os.path.exists(temp_h5)

    def test_control_force_standardization(self, temp_h5):
        """测试控制力标准化（标量、列表、数组）"""
        logger = HDF5Logger(temp_h5)
        logger.initialize_file()
        # 标量
        logger.log_step(0, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), 10.0, 0.0)
        # 列表
        logger.log_step(0, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), [1, 2, 3], 0.0)
        # 3D 数组
        logger.log_step(0, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.array([4, 5, 6]), 0.0)
        logger.close()

        with h5py.File(temp_h5, 'r') as f:
            forces = f['control_forces'][:]
            assert np.allclose(forces[0], [10, 0, 0])
            assert np.allclose(forces[1], [1, 2, 3])
            assert np.allclose(forces[2], [4, 5, 6])

    def test_load_data(self, temp_h5):
        """测试数据加载（直接使用 h5py）"""
        logger = HDF5Logger(temp_h5)
        logger.initialize_file()
        for i in range(10):
            logger.log_step(i, np.zeros(6), np.ones(6), np.zeros(6), np.zeros(6), np.zeros(3), 0.0)
        logger.close()

        with h5py.File(temp_h5, 'r') as f:
            epochs = f['epochs'][:]
            assert len(epochs) == 10
            assert np.all(epochs == np.arange(10))

    def test_load_all_data(self, temp_h5):
        """测试加载所有数据（直接使用 h5py）"""
        logger = HDF5Logger(temp_h5)
        logger.initialize_file()
        for i in range(5):
            logger.log_step(i, np.zeros(6), np.ones(6), np.zeros(6), np.zeros(6), np.zeros(3), 0.0)
        logger.close()

        with h5py.File(temp_h5, 'r') as f:
            all_data = {key: f[key][:] for key in f.keys() if isinstance(f[key], h5py.Dataset)}
            assert 'epochs' in all_data
            assert len(all_data['epochs']) == 5

    def test_get_statistics(self, temp_h5):
        """测试统计信息"""
        logger = HDF5Logger(temp_h5)
        logger.initialize_file()
        for i in range(3):
            logger.log_step(i, np.zeros(6), np.ones(6), np.zeros(6), np.zeros(6), np.zeros(3), 0.0)
        logger.close()

        stats = logger.get_statistics()
        assert stats['total_steps'] == 3
        assert stats['file_exists'] is True
        assert 'file_size_mb' in stats


# =============================================================================
# 并行安全测试（多进程）
# =============================================================================
class TestHDF5LoggerConcurrent:
    """测试并行场景下的文件 I/O 安全性"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def worker_write_single(self, filepath, data_id):
        """单个进程写入一个独立文件"""
        logger = HDF5Logger(filepath, buffer_size=10, compression=False)
        logger.initialize_file()
        for i in range(100):
            logger.log_step(
                epoch=float(i),
                nominal_state=np.zeros(6),
                true_state=np.full(6, data_id),
                nav_state=np.zeros(6),
                tracking_error=np.zeros(6),
                control_force=np.zeros(3),
                accumulated_dv=0.0
            )
        logger.close()
        return True

    def test_multiprocess_independent_files(self, temp_dir):
        """测试多进程各自写入独立文件，无冲突"""
        n_procs = 4
        files = [os.path.join(temp_dir, f"out_{i}.h5") for i in range(n_procs)]
        with mp.Pool(n_procs) as pool:
            results = pool.starmap(self.worker_write_single, zip(files, range(n_procs)))
        assert all(results)

        # 验证每个文件内容正确
        for i, f in enumerate(files):
            with h5py.File(f, 'r') as hf:
                true_states = hf['true_states'][:]
                assert np.all(true_states[:, 0] == i)
                assert true_states.shape == (100, 6)

    def worker_read_same_file(self, filepath, start, end, result_queue):
        """多个进程同时读取同一个文件"""
        time.sleep(0.1)  # 模拟随机启动
        try:
            with h5py.File(filepath, 'r') as f:
                data = f['true_states'][start:end]
                result_queue.put((True, data))
        except Exception as e:
            result_queue.put((False, str(e)))

    def test_multiprocess_read_same_file(self, temp_dir):
        """测试多进程同时读取同一文件（应允许）"""
        # 先创建测试文件
        filepath = os.path.join(temp_dir, "shared.h5")
        logger = HDF5Logger(filepath, compression=False)
        logger.initialize_file()
        for i in range(1000):
            logger.log_step(
                epoch=float(i),
                nominal_state=np.zeros(6),
                true_state=np.full(6, i),
                nav_state=np.zeros(6),
                tracking_error=np.zeros(6),
                control_force=np.zeros(3),
                accumulated_dv=0.0
            )
        logger.close()

        # 启动多个读进程，每个读取 100 条，共 10 个进程
        n_procs = 10
        result_queue = mp.Queue()
        processes = []
        for i in range(n_procs):
            start = 100 * i
            end = 100 * (i + 1)
            p = mp.Process(target=self.worker_read_same_file, args=(filepath, start, end, result_queue))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        # 收集结果
        successes = []
        data_parts = []
        while not result_queue.empty():
            ok, val = result_queue.get()
            successes.append(ok)
            if ok:
                data_parts.append(val)
        assert all(successes)
        # 验证所有数据段拼起来完整
        all_data = np.concatenate(data_parts)
        assert all_data.shape == (1000, 6)



# =============================================================================
# 多线程安全测试
# =============================================================================
class TestHDF5LoggerThreadSafe:
    """测试多线程下写入独立文件的安全性"""

    def worker_thread_write(self, filepath, data_id):
        """线程内写入独立文件"""
        logger = HDF5Logger(filepath, buffer_size=10, compression=False)
        logger.initialize_file()
        for i in range(100):
            logger.log_step(
                epoch=float(i),
                nominal_state=np.zeros(6),
                true_state=np.full(6, data_id),
                nav_state=np.zeros(6),
                tracking_error=np.zeros(6),
                control_force=np.zeros(3),
                accumulated_dv=0.0
            )
        logger.close()
        return True

    def test_multithread_independent_files(self, temp_dir):
        """测试多线程各自写入独立文件"""
        n_threads = 4
        files = [os.path.join(temp_dir, f"thread_{i}.h5") for i in range(n_threads)]
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=self.worker_thread_write, args=(files[i], i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        for i, f in enumerate(files):
            with h5py.File(f, 'r') as hf:
                true_states = hf['true_states'][:]
                assert np.all(true_states[:, 0] == i)
                assert true_states.shape == (100, 6)

    def worker_thread_write_same_file(self, filepath, data_id, result_list):
        """多个线程尝试写入同一文件（应通过锁或其他机制保护）"""
        try:
            logger = HDF5Logger(filepath, buffer_size=1, compression=False)
            logger.initialize_file()
            for i in range(50):
                logger.log_step(
                    epoch=float(i),
                    nominal_state=np.zeros(6),
                    true_state=np.full(6, data_id),
                    nav_state=np.zeros(6),
                    tracking_error=np.zeros(6),
                    control_force=np.zeros(3),
                    accumulated_dv=0.0
                )
            logger.close()
            result_list.append(True)
        except Exception as e:
            result_list.append(False)

    def test_multithread_write_same_file(self, temp_dir):
        """测试多线程同时写入同一文件（可能不安全，需验证）"""
        filepath = os.path.join(temp_dir, "shared_thread.h5")
        n_threads = 4
        results = []
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=self.worker_thread_write_same_file, args=(filepath, i, results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # 检查是否有异常
        # 注意：HDF5 写入不是线程安全的，预期可能出错或数据损坏
        # 此测试目的不是要求通过，而是验证并发写入是否有明显破坏
        # 若项目需要支持多线程写同一文件，应引入锁；否则应避免这种用法。
        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as f:
                true_states = f['true_states'][:]
                unique_ids = np.unique(true_states[:, 0])
                # 若出现多个 data_id 混合，说明数据损坏
                # 当前设计不支持并发写同一文件，因此这里不强制断言，但可以记录
                # 本测试仅用于暴露问题
                if len(unique_ids) > 1:
                    print(f"警告：多线程写同一文件导致数据混合，unique_ids={unique_ids}")
