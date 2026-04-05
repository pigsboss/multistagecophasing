"""
Sprint 2 Acceptance Test Script

一键验收 Sprint 2 出口标准：
1. ✅ 打靶法能收敛到 2:1 共振轨道
2. ✅ 最终位置残差 < 1e-6
3. ✅ 雅可比常数在积分过程中保持守恒（漂移 < 1e-8）
4. ✅ 周期一致性验证（重积分后残差 < 1e-5）

用法：
    python acceptance_sprint2.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


def check_jacobi_conservation(targeter, initial_state, period):
    """验证雅可比常数守恒"""
    mu = targeter.mu
    num_points = 100
    t_span = np.linspace(0, period / (4.342 * 86400), num_points)
    
    jacobi_history = []
    dt = t_span[1] - t_span[0]
    x = initial_state.copy()
    dynamics = targeter._get_dynamics_func()
    
    for i in range(len(t_span)):
        r1_sq = (x[0] + mu)**2 + x[1]**2 + x[2]**2
        r2_sq = (x[0] + mu - 1)**2 + x[1]**2 + x[2]**2
        r1 = np.sqrt(max(r1_sq, 1e-10))
        r2 = np.sqrt(max(r2_sq, 1e-10))
        v_sq = x[3]**2 + x[4]**2 + x[5]**2
        C = x[0]**2 + x[1]**2 + 2*(1-mu)/r1 + 2*mu/r2 - v_sq
        jacobi_history.append(C)
        
        if i < len(t_span) - 1:
            k1 = dynamics(t_span[i], x)
            k2 = dynamics(t_span[i] + 0.5*dt, x + 0.5*dt*k1)
            k3 = dynamics(t_span[i] + 0.5*dt, x + 0.5*dt*k2)
            k4 = dynamics(t_span[i] + dt, x + dt*k3)
            x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    C0 = jacobi_history[0]
    max_rel_change = max([abs((C - C0) / C0) for C in jacobi_history])
    return max_rel_change, jacobi_history


def main():
    print("=" * 70)
    print("LUNAR-SWING Sprint 2 Acceptance Test")
    print("=" * 70)
    print()
    
    # Setup
    print("1. Initializing CRTBP and Targeter...")
    crtbp = UniversalCRTBP.earth_moon_system()
    targeter = LunarSwingTargeter(
        dynamics_model=crtbp,
        mu=crtbp.mu,
        integrator_type='rk4',
        num_steps=1000  # Increase for better accuracy
    )
    print("   ✅ CRTBP and Targeter initialized")
    print(f"   System: {crtbp.system_name}")
    print(f"   μ = {crtbp.mu:.6f}")
    print()
    
    # 2:1 Resonance initial guess (known to converge)
    resonance_n, resonance_m = 2, 1
    initial_guess = np.array([0.95, 0.0, 0.0, 0.0, 1.15, 0.0])
    tol = 1e-6
    max_iter = 100
    damping = 0.8
    
    print(f"2. Searching for {resonance_n}:{resonance_m} resonant orbit...")
    print(f"   Initial guess: {initial_guess}")
    print(f"   Tolerance: {tol}")
    print(f"   Max iterations: {max_iter}")
    print(f"   Damping: {damping}")
    print()
    
    # Run search
    result = targeter.find_resonant_orbit(
        resonance_ratio=(resonance_n, resonance_m),
        initial_guess=initial_guess,
        tol=tol,
        max_iter=max_iter,
        damping=damping
    )
    
    # Collect results
    history = result['convergence_history']
    final_residual = history[-1]['residual_norm']
    num_iterations = len(history)
    
    # Acceptance criteria
    criteria = {
        'converged': result['success'],
        'residual_ok': final_residual < tol,
        'jacobi_ok': False,  # To be checked
        'period_ok': False   # To be checked
    }
    
    print("3. Checking acceptance criteria...")
    print()
    
    # Criterion 1 & 2: Convergence and residual
    print(f"   Criterion 1: Search converged?")
    print(f"      {'✅ PASS' if criteria['converged'] else '❌ FAIL'} - {'Converged' if result['success'] else 'Not converged'}")
    print()
    
    print(f"   Criterion 2: Final residual < {tol}?")
    print(f"      {'✅ PASS' if criteria['residual_ok'] else '❌ FAIL'} - Residual = {final_residual:.2e}")
    print(f"      Iterations: {num_iterations}")
    if len(history) > 1:
        improvement = history[0]['residual_norm'] / history[-1]['residual_norm']
        print(f"      Improvement: {improvement:.1f}x")
    print()
    
    # Criterion 3: Jacobi constant conservation
    if criteria['converged']:
        print(f"   Criterion 3: Jacobi constant conservation (drift < 1e-8)?")
        converged_state = result['state']
        period = result['period']
        max_jacobi_drift, _ = check_jacobi_conservation(targeter, converged_state, period)
        criteria['jacobi_ok'] = max_jacobi_drift < 1e-8
        print(f"      {'✅ PASS' if criteria['jacobi_ok'] else '❌ FAIL'} - Max drift = {max_jacobi_drift:.2e}")
    else:
        print(f"   Criterion 3: Jacobi constant conservation (drift < 1e-8)?")
        print(f"      ⚠️ SKIP - Search did not converge")
    print()
    
    # Criterion 4: Period consistency recheck
    if criteria['converged']:
        print(f"   Criterion 4: Period consistency (recheck residual < 1e-5)?")
        try:
            x_final, _ = targeter._stm_calc.propagate_with_stm(
                dynamics=targeter._get_dynamics_func(),
                initial_state=converged_state,
                t0=0.0,
                tf=period / (4.342 * 86400),
                method=targeter.integrator_type,
                num_steps=targeter.num_steps
            )
            recheck_residual = np.linalg.norm(x_final - converged_state)
            criteria['period_ok'] = recheck_residual < 1e-5
            print(f"      {'✅ PASS' if criteria['period_ok'] else '❌ FAIL'} - Recheck residual = {recheck_residual:.2e}")
        except Exception as e:
            print(f"      ⚠️ ERROR - {e}")
    else:
        print(f"   Criterion 4: Period consistency (recheck residual < 1e-5)?")
        print(f"      ⚠️ SKIP - Search did not converge")
    print()
    
    # Overall result
    print("=" * 70)
    print("Overall Result")
    print("=" * 70)
    
    all_passed = all(criteria.values())
    
    if all_passed:
        print("🎉🎉🎉 Sprint 2 Acceptance: ✅ PASSED 🎉🎉🎉")
    else:
        print("⚠️ Sprint 2 Acceptance: ❌ FAILED ⚠️")
    
    print()
    print("Summary:")
    for name, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    print()
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
