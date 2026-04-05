"""
Resonant Orbit Search and Visualization Tool

Searches for Earth-Moon resonant periodic orbits using single-parameter shooting
and generates visualization plots:
- 3D trajectory plot (Earth-Moon rotating frame)
- Poincare section (y=0 plane)
- Energy conservation (Jacobi constant history)
- Convergence history

Usage Examples:
    # Default 2:1 resonance search
    python search_and_visualize_resonant_orbit.py
    
    # 3:2 resonance with custom parameters
    python search_and_visualize_resonant_orbit.py --resonance-ratio 3:2 --damping 0.6
    
    # High precision search
    python search_and_visualize_resonant_orbit.py --tol 1e-8 --max-iter 200
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path (handle both direct run and module import)
try:
    from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
except ImportError:
    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


def generate_3d_trajectory(targeter, initial_state, period, output_path):
    """Generate 3D trajectory plot in Earth-Moon rotating frame."""
    num_points = 500
    t_span = np.linspace(0, period / (4.342 * 86400), num_points)
    
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
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(states[:, 0], states[:, 1], states[:, 2], 
            'b-', linewidth=1.5, label='Orbit')
    ax.plot([states[0, 0]], [states[0, 1]], [states[0, 2]], 
            'go', markersize=8, label='Start/End')
    
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
    num_periods = 10
    total_time = period * num_periods / (4.342 * 86400)
    
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
    
    crossings_x = []
    crossings_vx = []
    
    for i in range(len(states)-1):
        if states[i, 1] * states[i+1, 1] < 0:
            t_cross = abs(states[i, 1]) / (abs(states[i, 1]) + abs(states[i+1, 1]))
            x_cross = states[i, 0] + t_cross * (states[i+1, 0] - states[i, 0])
            vx_cross = states[i, 3] + t_cross * (states[i+1, 3] - states[i, 3])
            crossings_x.append(x_cross)
            crossings_vx.append(vx_cross)
    
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
            x = np.clip(x, -1e4, 1e4)
    
    t_hours = t_span * 4.342 * 24
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t_hours, jacobi_history, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Jacobi Constant C')
    ax1.set_title('Jacobi Constant Conservation')
    ax1.grid(True, alpha=0.3)
    
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
    parser = argparse.ArgumentParser(
        description='Search for resonant lunar swing orbit and generate visualization plots.'
    )
    parser.add_argument('--resonance-ratio', type=str, default='2:1',
                        help='Target resonance ratio n:m (default: 2:1)')
    parser.add_argument('--damping', type=float, default=0.8,
                        help='Damping factor for Newton iteration (0-1, default: 0.8)')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: ./orbit_plots)')
    
    args = parser.parse_args()
    
    # Parse resonance ratio
    try:
        n, m = map(int, args.resonance_ratio.split(':'))
    except ValueError:
        print(f"Error: Invalid resonance ratio format '{args.resonance_ratio}'. Use n:m format (e.g., 2:1)")
        return
    
    print("=" * 60)
    print("Resonant Orbit Search and Visualization")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Resonance ratio: {n}:{m}")
    print(f"  Damping: {args.damping}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Tolerance: {args.tol}")
    print("-" * 60)
    
    # Setup
    crtbp = UniversalCRTBP.earth_moon_system()
    targeter = LunarSwingTargeter(
        dynamics_model=crtbp,
        mu=crtbp.mu,
        integrator_type='rk4',
        num_steps=200
    )
    
    # Output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "orbit_plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Initial guess (adjust based on resonance)
    mu = targeter.mu
    if (n, m) == (1, 1):
        # L1 family orbit - better initial guess based on linearized theory
        L1_x = 0.836915
        # vy ≈ sqrt(3*mu) for small amplitude L1 orbits
        vy_guess = np.sqrt(3 * mu) * 0.8
        initial_guess = np.array([L1_x + 0.02, 0.0, 0.0, 0.0, vy_guess, 0.0])
        print(f"Using L1-family initial guess: x={initial_guess[0]:.4f}, vy={vy_guess:.4f}")
    elif (n, m) == (2, 1):
        # 2:1 resonance - high eccentricity orbit
        # Apogee near Moon (x ~ 1), perigee near Earth (x ~ -mu)
        initial_guess = np.array([0.95, 0.0, 0.0, 0.0, 1.15, 0.0])
        print(f"Using 2:1 resonance initial guess: x={initial_guess[0]:.4f}, vy={initial_guess[4]:.4f}")
    else:
        # Generic guess - start from circular-ish orbit
        a_resonant = ((m/n) ** (2/3))  # Semi-major axis for resonance
        x_guess = a_resonant - mu
        vy_guess = np.sqrt((1-mu)/abs(x_guess + mu)) * 0.9
        initial_guess = np.array([x_guess, 0.0, 0.0, 0.0, vy_guess, 0.0])
        print(f"Using generic initial guess for {n}:{m}: x={x_guess:.4f}, vy={vy_guess:.4f}")
    
    # Run search
    print(f"Searching for {n}:{m} resonant orbit...")
    result = targeter.find_resonant_orbit(
        resonance_ratio=(n, m),
        initial_guess=initial_guess,
        tol=args.tol,
        max_iter=args.max_iter,
        damping=args.damping
    )
    
    print(f"Search completed: {'Converged' if result['success'] else 'Not converged'}")
    print(f"Final residual: {result['convergence_history'][-1]['residual_norm']:.2e}")
    
    # Print convergence stats
    history = result['convergence_history']
    if len(history) > 1:
        improvement = history[0]['residual_norm'] / history[-1]['residual_norm']
        print(f"Residual improvement: {improvement:.1f}x")
    print()
    
    if result['success']:
        converged_state = result['state']
        period = result['period']
    else:
        converged_state = result['state']
        period = result['period']
        print("Warning: Using best available state for visualization")
    
    # Generate plots
    print("Generating visualizations...")
    print("-" * 40)
    
    generate_3d_trajectory(
        targeter, converged_state, period,
        output_dir / "orbit_3d_trajectory.png"
    )
    
    generate_poincare_section(
        targeter, converged_state, period,
        output_dir / "poincare_section.png"
    )
    
    generate_energy_conservation(
        targeter, converged_state, period,
        output_dir / "energy_conservation.png"
    )
    
    # Convergence history
    iterations = [h['iteration'] for h in history]
    residuals = [h['residual_norm'] for h in history]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(iterations, residuals, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position Residual Norm (log)')
    ax.set_title(f'Shooting Method Convergence\n{"Converged" if result["success"] else "Not Converged"}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=args.tol, color='r', linestyle='--', 
               label=f'Convergence Threshold ({args.tol})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence history saved to: {output_dir / 'convergence_history.png'}")
    print("-" * 40)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
