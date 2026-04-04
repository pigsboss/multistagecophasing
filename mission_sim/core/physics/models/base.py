"""
DEPRECATED: This file has been removed as part of the ForceModel abstraction cleanup.
All force models should directly implement the IForceModel interface from
mission_sim.core.physics.environment.

To update your code:
1. Import IForceModel from mission_sim.core.physics.environment
2. Make your force model class inherit directly from IForceModel
3. Implement the required methods: compute_accel and optionally compute_vectorized_acc

Example:
    from mission_sim.core.physics.environment import IForceModel
    
    class MyForceModel(IForceModel):
        def compute_accel(self, state, epoch):
            # Your implementation
            pass
"""

raise ImportError(
    "ForceModel base class has been removed. "
    "Please update your imports to use IForceModel directly from "
    "mission_sim.core.physics.environment"
)
