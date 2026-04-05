"""
Stage 2 Visualization Generator for Lunar Swing Orbit

Generates required visualizations for Stage 2 exit criteria:
- 3D trajectory plot (Earth-Moon rotating frame)
- Poincare section (y=0 plane)
- Energy conservation (Jacobi constant history)
- Convergence history

Usage:
    python generate_stage2_visualizations.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


def generate_3d_trajectory(targeter, initial_state, period, output_path):
    """Generate 3D trajectory plot in Earth-Moon rotating frame."""
    # Propagate orbit with dense output for smooth trajectory
    num_points = 500
    t_span = np.linspace(0, period / (4.342 * 86400), num_points)  # Nondimensional time
    
    states = [initial_state.copy()]
    dt = t_span[1] - t_span[0]
    x = initial_state.copy()
    
    dynamics = targeter._get_dynamics_func()
    
    for i in range(1, len(t_span)):
        # RK4 step
        k1 = dynamics(t_span[i-1], x)
        k2 = dynamics(t_span[i-1] + 0.5*dt, x + 0.5*dt*k1)
        k3 = dynamics(t_span[i-1] + 0.5*dt, x + 0.5*dt*k2)
        k4 = dynamics(t_span[i-1] + dt, x + dt*k3)
        x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x = np.clip(x, -1e4, 1e4)  # Prevent overflow
        states.append(x.copy())
    
    states = np.array(states)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 
            'b-', linewidth=1.5, label='Orbit')
    
    # Mark start/end point
    ax.plot([states[0, 0]], [states[0, 1]], [states[0, 2]], 
            'go', markersize=8, label='Start/End')
    
    # Mark Earth and Moon positions
    mu = targeter.mu
    ax.plot([-mu], [0], [0], 'b^', markersize=10, label='Earth')
    ax.plot([1-mu], [0], [0], 'r^', markersize=8, label='Moon')
    
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized)')
    ax.set_title(f'Resonant Orbit 3D Trajectory\nPeriod: {period/86400:.2f} days')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D trajectory saved to: {output_path}")
    return states


def generate_poincare_section(targeter, initial_state, period, output_path):
    """Generate Poincare section plot (y=0 plane crossings)."""
    # Long propagation to get many crossings
    num_periods = 10
    total_time = period * num_periods / (4.342 * 86400)  # Nondimensional
    
    num_points = 2000
    t_span = np.linspace(0, total_time, num_points)
    
    states = [initial_state.copy()]
    dt = t_span[1] - t_span[0]
    x = initial_state.copy()
    
    dynamics = targeter._get_dynamics_func()
    
    for i in range(1, len(t_span)):
        k1 = dynamics(t_span[i-1], x)
        k2 = dynamics(t_span[i-1] + 0.5*dt, x + 0.5*dt*k1)
        k3 = dynamics(t_span[i-1] + 0.5*dt, x + 0.5*dt*k2)
        k4 = dynamics(t_span[i-1] + dt, x + dt*k3)
        x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x = np.clip(x, -1e4, 1e4)
        states.append(x.copy())
    
    states = np.array(states)
    
    # Find y=0 crossings
    crossings_x = []
    crossings_vx = []
    
    for i in range(len(states)-1):
        if states[i, 1] * states[i+1, 1] < 0:  # Sign change
            # Linear interpolation for better accuracy
            t_cross = abs(states[i, 1]) / (abs(states[i, 1]) + abs(states[i+1, 1]))
            x_cross = states[i, 0] + t_cross * (states[i+1, 0] - states[i, 0])
            vx_cross = states[i, 3] + t_cross * (states[i+1, 3] - states[i, 3])
            crossings_x.append(x_cross)
            crossings_vx.append(vx_cross)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(crossings_x) > 0:
        ax.scatter(crossings_x, crossings_vx, c='r', s=20, alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('vx')
        ax.set_title(f'Poincare Section (y=0 plane)\n{len(crossings_x)} crossings')
    else:
        ax.text(0.5, 0.5, 'No crossings found', ha='center', va='center')
        ax.set_title('Poincare Section (y=0 plane)')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Poincare section saved to: {output_path}")


def generate_energy_conservation(targeter, initial_state, period, output_path):
    """Generate energy conservation (Jacobi constant) history."""
    mu = targeter.mu
    
    num_points = 200
    t_span = np.linspace(0, period / (4.342 * 86400), num_points)
    
    jacobi_history = []
    
    dt = t_span[1] - t_span[0]
    x = initial_state.copy()
    
    dynamics = targeter._get_dynamics_func()
    
    for i in range(len(t_span)):
        # Compute Jacobi constant
        r1_sq = (x[0] + mu)**2 + x[1]**2 + x[2]**2
        r2_sq = (x[0] + mu - 1)**2 + x[1]**2 + x[2]**2
        
        r1 = np.sqrt(max(r1_sq, 1e-10))
        r2 = np.sqrt(max(r2_sq, 1e-10))
        
        # C = x^2 + y^2 + 2(1-mu)/r1 + 2*mu/r2 - v^2
        v_sq = x[3]**2 + x[4]**2 + x[5]**2
        C = x[0]**2 + x[1]**2 + 2*(1-mu)/r1 + 2*mu/r2 - v_sq
        
        jacobi_history.append(C)
        
        if i < len(t_span) - 1:
            k1 = dynamics(t_span[i], x)
            k2 = dynamics(t_span[i] + 0.5*dt, x + 0.5*dt*k1)
            k3 = dynamics(t_span[i] + 0.5*dt, x + 0.5*dt*k2)
            k4 = dynamics(t_span[i] + dt, x + dt*k3)
            x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x = np.clip(x, -1e4, 1e4)
    
    t_hours = t_span * 4.342 * 24  # Convert to hours
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Jacobi constant history
    ax1.plot(t_hours, jacobi_history, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Jacobi Constant C')
    ax1.set_title('Jacobi Constant Conservation')
    ax1.grid(True, alpha=0.3)
    
    # Relative change
    C0 = jacobi_history[0]
    rel_change = [(C - C0)/abs(C0) for C in jacobi_history]
    ax2.semilogy(t_hours[1:], np.abs(rel_change)[1:], 'r-', linewidth=1.5)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Relative Change |dC/C|')
    ax2.set_title(f'Energy Drift (final: {rel_change[-1]:.2e})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Energy conservation plot saved to: {output_path}")
    print(f"Final energy drift: {rel_change[-1]:.2e}")


def main():
    """Main function to generate all Stage 2 visualizations."""
    print("=" * 60)
    print("Stage 2 Visualization Generator")
    print("=" * 60)
    
    # Setup
    crtbp = UniversalCRTBP.earth_moon_system()
    targeter = LunarSwingTargeter(
        dynamics_model=crtbp,
        mu=crtbp.mu,
        integrator_type='rk4',
        num_steps=200
    )
    
    # Create output directory
    output_dir = Path(__file__).parent / "stage2_outputs"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Initial guess for 2:1 resonance orbit
    mu = targeter.mu
    L1_x = 0.8369
    
    initial_guess = np.array([
        L1_x + 0.08, 0.0, 0.0,
        0.0, 0.35, 0.0
    ])
    
    # Run resonant orbit search
    print("Searching for 2:1 resonant orbit...")
    result = targeter.find_resonant_orbit(
        resonance_ratio=(2, 1),
        initial_guess=initial_guess,
        tol=1e-5,
        max_iter=50,
        damping=0.5
    )
    
    print(f"Search completed: {'Converged' if result['success'] else 'Not converged'}")
    print(f"Final residual: {result['convergence_history'][-1]['residual_norm']:.2e}")
    print()
    
    if result['success']:
        converged_state = result['state']
        period = result['period']
    else:
        # Use best available state
        converged_state = result['state']
        period = result['period']
        print("Using best available state for visualization")
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 40)
    
    # 1. 3D Trajectory
    generate_3d_trajectory(
        targeter, converged_state, period,
        output_dir / "orbit_3d_trajectory.png"
    )
    
    # 2. Poincare Section
    generate_poincare_section(
        targeter, converged_state, period,
        output_dir / "poincare_section.png"
    )
    
    # 3. Energy Conservation
    generate_energy_conservation(
        targeter, converged_state, period,
        output_dir / "energy_conservation.png"
    )
    
    # 4. Convergence History (already generated by test, copy or regenerate)
    history = result['convergence_history']
    iterations = [h['iteration'] for h in history]
    residuals = [h['residual_norm'] for h in history]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(iterations, residuals, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position Residual Norm (log)')
    ax.set_title(f'Shooting Method Convergence\n{"Converged" if result["success"] else "Not Converged"}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-5, color='r', linestyle='--', label='Convergence Threshold (1e-5)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence history saved to: {output_dir / 'convergence_history.png'}")
    print("-" * 40)
    print()
    print("All visualizations generated successfully!")
    print(f"Output files in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
