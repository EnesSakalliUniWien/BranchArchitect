
import numpy as np

def simulate_trajectory(rng, targets, n_points, n_divergence_steps, n_resolution_steps, linear_frac, endpoint_noise_std):
    """Generate two-phase motion trajectory."""
    # Motion modes: 0=linear, 1=ballistic
    modes = rng.choice([0, 1], size=n_points, p=[linear_frac, 1 - linear_frac])

    # Phase 1: Divergence trajectory
    T_diverge = n_divergence_steps - 1

    if T_diverge <= 0: # Handle n_divergence_steps = 1 or less
        divergence_traj = targets[:, np.newaxis, :] # All points immediately at target
    else:
        v0 = rng.uniform(-0.2, 0.2, size=(n_points, 3))
        acc = np.zeros_like(v0)
        ball = modes == 1

        acc[ball] = 2 * (targets[ball] - v0[ball] * T_diverge) / (T_diverge**2)

        t_vec_diverge = np.arange(n_divergence_steps).reshape(1, n_divergence_steps, 1)

        linear_traj = targets[:, np.newaxis, :] * (t_vec_diverge / T_diverge)

        ballistic_traj = (v0[:, np.newaxis, :] * t_vec_diverge) + (
            0.5 * acc[:, np.newaxis, :] * t_vec_diverge**2
        )
        divergence_traj = np.where(
            (modes == 1).reshape(n_points, 1, 1), ballistic_traj, linear_traj
        )

    # Phase 2: Resolution trajectory with gradual noise
    start_pos_resolution = divergence_traj[:, -1, :]
    final_noise = rng.normal(0, endpoint_noise_std, size=(n_points, 3))
    noise_scaling = np.linspace(0, 1, n_resolution_steps)
    resolution_traj = start_pos_resolution[:, np.newaxis, :] + (
        final_noise[:, np.newaxis, :] * noise_scaling[np.newaxis, :, np.newaxis]
    )

    # Combine trajectories
    full_traj = np.concatenate((divergence_traj, resolution_traj[:, 1:, :]), axis=1)

    return full_traj, modes
