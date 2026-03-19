import os
import numpy as np

# 导入核心域：物理实体与信息系统
from core.types import CoordinateFrame
from core.spacecraft import SpacecraftPointMass
from core.environment import CelestialEnvironment
from core.ground_station import GroundStation
from core.gnc_subsystem import GNC_Subsystem

# 导入工具域：持久化记录与可视化
from utils.loggers import HDF5Logger
from utils.visualizer import L1Visualizer

def main():
    print("="*60)
    print("🚀 [Level 1] 单航天器平动点轨道维持仿真 - 启动")
    print("="*60)
    
    # --- 1. 仿真全局配置 ---
    dt = 1.0                # 动力学积分步长 (秒)
    total_time = 3600 * 2   # 仿真总时长 (2小时)
    steps = int(total_time / dt)
    log_interval = 10       # 每 10 步落盘一次数据
    
    # 确保输出目录存在
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    h5_filepath = os.path.join(out_dir, "L1_simulation.h5")
    
    # 全局强制契约：日地旋转系
    MISSION_FRAME = CoordinateFrame.SUN_EARTH_ROTATING
    
    # --- 2. 实例化系统组件 ---
    env = CelestialEnvironment(region="SUN_EARTH_L2", initial_epoch=0.0)
    
    # 目标标称点 (L2 理论位置)
    X_E = (1.0 - env.mu) * env.AU
    X_L2 = X_E + 1.508e9
    target_state = np.array([X_L2, 0.0, 0.0, 0.0, 178.5, 0.0], dtype=np.float64)
    
    # 航天器初始状态 (故意在 3D 空间中制造初始偏差，便于观察 GNC 动态响应)
    initial_offset = np.array([1000.0, -500.0, 200.0, 0.0, 0.0, 0.0])
    initial_state = target_state + initial_offset
    
    sc = SpacecraftPointMass(sc_id="Probe-Alpha", initial_state=initial_state, frame=MISSION_FRAME)
    gs = GroundStation(name="DeepSpace_Net", operating_frame=MISSION_FRAME, pos_noise_std=2.0, vel_noise_std=0.001)
    gnc = GNC_Subsystem(sc_id=sc.id, operating_frame=MISSION_FRAME)
    
    # 简单的 LQR/PD 三轴反馈增益矩阵
    K_LQR = np.zeros((3, 6))
    K_LQR[0,0], K_LQR[1,1], K_LQR[2,2] = 0.005, 0.005, 0.005  # 位置刚度
    K_LQR[0,3], K_LQR[1,4], K_LQR[2,5] = 0.8, 0.8, 0.8        # 速度阻尼

    # --- 3. 核心推演循环 (集成 HDF5 增量记录) ---
    print(f"[*] 开启主控循环，共计 {steps} 步，数据将增量存入: {h5_filepath}")
    
    with HDF5Logger(h5_filepath, flush_interval=100) as logger:
        # 将坐标系契约作为元数据永久刻印在 HDF5 文件中
        logger.set_metadata(path=f"/{sc.id}", meta_dict={"frame": MISSION_FRAME})
        
        for i in range(steps):
            t = i * dt
            
            # --- 信息域：测控与指令 ---
            obs_state, obs_frame = gs.track_spacecraft(sc.state, sc.frame)
            cmd_packet = gs.generate_telecommand("ORBIT_MAINTENANCE", target_state, MISSION_FRAME)
            
            # --- 计算域：GNC 算法 ---
            gnc.process_telecommand(cmd_packet)
            gnc.update_navigation(obs_state, obs_frame)
            force_cmd, force_frame = gnc.compute_control_force(K_LQR)
            
            # --- 物理域：动力学推演 ---
            sc.apply_thrust(force_cmd, force_frame)
            grav_accel, grav_frame = env.get_gravity_acceleration(sc.state, sc.frame)
            derivative = sc.get_derivative(grav_accel, grav_frame)
            
            # Euler 积分推进
            sc.state += derivative * dt  
            sc.consume_mass(m_dot=0.0001, dt=dt) 
            
            # 状态重置与时间推移
            sc.clear_thrust()
            env.step_time(dt) 
            
            # --- 数据高频采样落盘 ---
            if i % log_interval == 0:
                logger.log(sc.id, "time", t)
                logger.log(sc.id, "true_state", sc.state)
                logger.log(sc.id, "target_state", target_state)
                logger.log(sc.id, "thrust", force_cmd)
                
            # 通知 logger 时间步推移，触发缓存自动 flush
            logger.step()

    print("[*] 物理推演结束。所有高维张量已安全落盘。")
    print("-" * 60)
    
    # --- 4. 离线数据可视化 (读取 HDF5 进行后处理) ---
    print("📊 正在启动数据可视化引擎...")
    vis = L1Visualizer(h5_filepath, sc.id)
    
    # 1. 状态时序曲线
    state_plot_path = os.path.join(out_dir, "L1_state_history.png")
    vis.plot_state_history(save_path=state_plot_path)
    
    # 2. GNC 推力活动时序
    gnc_plot_path = os.path.join(out_dir, "L1_gnc_activity.png")
    vis.plot_gnc_activity(save_path=gnc_plot_path)
    
    # 3. 带有矢量推力的 3D 轨迹动画
    # 注意: mp4 渲染需要系统级安装 ffmpeg。如果遇到报错，可将后缀改为 .gif
    anim_plot_path = os.path.join(out_dir, "L1_trajectory.mp4") 
    vis.create_animation(save_path=anim_plot_path, downsample=10, thrust_scale=30.0)

    print("="*60)
    print("✅ Level 1 闭环仿真全流程执行完毕！所有产物已存放于 output/ 目录。")
    print("="*60)

if __name__ == "__main__":
    main()
