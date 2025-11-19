import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from subbots_sim.config import global_vars
from subbots_sim.analysis.math_tools import cylindrical_to_xy
from subbots_sim.sim.simulate_signals_for_pinger import (
    simulate_continous_signals_for_pinger,
    simulate_pulsed_signals_for_pinger,
    plot_hydrophone_signal
)
from subbots_sim.analysis.position_estimation import localize_pinger_from_signals


def draw_ping_scene(
    hydrophone_positions,
    pinger_pos,
    center,
    view_radius,
    est_dir_angle=None,  
    est_dir_len=None,
    ax=None,
):
    """
    Plot a 2D diagram showing:
      - hydrophone array positions
      - true pinger position
      - array center
      - optional estimated direction (bearing)

    Inputs:
        hydrophone_positions : array (N,2)
        pinger_pos           : (2,) true pinger location
        center               : (2,) center of the hydrophone array
        view_radius          : float, how large of a window to plot
        est_dir_angle        : float or None, bearing estimate in radians
        est_dir_len          : float or None, length of the bearing line
    """
    # Create figure/axis if not provided
    if ax is None:
        fig, ax = plt.subplots()

    ax.clear()

    # Make sure everything is numpy arrays for easy indexing
    hydrophone_positions = np.asarray(hydrophone_positions)
    center = np.asarray(center)
    pinger_pos = np.asarray(pinger_pos)

    # Plot Hydrophones
    ax.scatter(
        hydrophone_positions[:, 0],
        hydrophone_positions[:, 1],
        marker="o",
        s=50,
        label="Hydrophones",
    )

    # Plot true pinger position
    ax.scatter(
        [pinger_pos[0]],
        [pinger_pos[1]],
        marker="x",
        s=100,
        linewidths=2,
        label="True pinger",
    )

    # PLot array center
    ax.scatter(
        [center[0]],
        [center[1]],
        marker="+",
        s=50,
        label="Array center",
    )

    # Estimated pinger direction
    # This is drawn as a faint dashed line from the center outward
    if est_dir_angle is not None:
        # If no length provided, just use view radius
        if est_dir_len is None:
            est_dir_len = view_radius

        # Convert angle to a direction vector
        dx = est_dir_len * np.cos(est_dir_angle)
        dy = est_dir_len * np.sin(est_dir_angle)

        ax.plot(
            [center[0], center[0] + dx],
            [center[1], center[1] + dy],
            linestyle="--",
            linewidth=1.5,
            alpha=0.4,
            label="Estimated bearing",
        )

    # Cosmetics
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Fix view around the center
    ax.set_xlim(center[0] - view_radius, center[0] + view_radius)
    ax.set_ylim(center[1] - view_radius, center[1] + view_radius)

    plt.draw()




def run_sim_without_analysis(num_trials=20):
    """
    Simple simulator: place the pinger at random angles on a fixed circle
    around the hydrophone array and plot it. No analysis yet.
    """
    hydrophones_xy = cylindrical_to_xy(global_vars.hydrophone_positions)
    center = np.array([0.0, 0.0])

    # tweak these to control the view
    PINGER_RADIUS = 0.20   # m from center
    VIEW_RADIUS   = 0.30   # axis half-width

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()

    for _ in range(num_trials):
        theta_true = rng.uniform(0, 2 * np.pi)
        pinger = center + PINGER_RADIUS * np.array([np.cos(theta_true), np.sin(theta_true)])

        draw_ping_scene(
            hydrophone_positions=hydrophones_xy,
            pinger_pos=pinger,
            center=center,
            view_radius=VIEW_RADIUS,
            est_dir_angle=None,   # no estimate here
            ax=ax,
        )

        plt.pause(2.0)

    plt.show()


def run_sim_with_localization(
    num_trials=20,
    hydrophone_z=-0.2,
    pinger_z=0.0,
    max_iterations=50,
    tol_steps=1e-6,
    tol_cost=1e-15,
    gamma=0.0,
):
    """
    Run multiple trials of:
        1) picking a random pinger position,
        2) simulating hydrophone signals,
        3) running Gauss-Newton localization,
        4) animating how the estimate moves each iteration.

    The animation shows:
        - hydrophones
        - true pinger
        - array center
        - estimated bearing at each GN iteration
    """

    # Convert hydrophone positions to XY for plotting
    hydrophones_xy = cylindrical_to_xy(global_vars.hydrophone_positions)

    # Define the center of the hydrophone array
    center = np.array([0.0, 0.0])

    # The circle radius for true pinger placement
    PINGER_RADIUS = 0.20

    # Plot view radius
    VIEW_RADIUS   = 0.5


    # Random number generator for pinger positions
    rng = np.random.default_rng(0) # Change number value for different sequences

    # Create figure/axis for plotting
    fig, ax = plt.subplots()

    # Run localization trials
    for trial in range(num_trials):
        #  Choose a random true pinger position on the circle
        theta_true = rng.uniform(0, 2 * np.pi)
        pinger_true = center + PINGER_RADIUS * np.array(
            [np.cos(theta_true), np.sin(theta_true)]
        )

        # Simulate hydrophone signals for this pinger position
        # For continous wave:
        """
        signals = simulate_continous_signals_for_pinger(
            pinger_xy=pinger_true,
            hydrophone_z=hydrophone_z,
            pinger_z=pinger_z,
            num_periods=200, 
            noise_std=0.05, 
        )
        """
        # For pulsed wave:
        signals = simulate_pulsed_signals_for_pinger(
            pinger_xy=pinger_true,
            hydrophone_z=hydrophone_z,
            pinger_z=pinger_z,
            num_periods=200,
            noise_std=0.05,
        )

        # View the signal reaching hydrophone 1 only at the beginning
        if trial == 0:
            plot_hydrophone_signal(
                signals,
               hydrophone_index=1,
               title_prefix="Hydrophone 1 signal"
            )

        #  Run Gauss-Newton localization and get its history
        # Make an initial guess slightly offset from center
        initial_guess = center + np.array([0.01, 0.0])


        pinger_est, history = localize_pinger_from_signals(
            initial_pinger_pos=initial_guess,
            hydrophone_z=hydrophone_z,
            pinger_z=pinger_z,
            signals=signals,
            max_iterations=max_iterations,
            tol_steps=tol_steps,
            tol_cost=tol_cost,
            gamma=gamma,
            max_step_norm=0.05,
            pinger_radius=PINGER_RADIUS,   # constrain to circle 
        )


        # List of positions from each GN iteration
        positions_over_iters = history["positions"]

        # Animate the GN iterations
        for iter_idx, pos in enumerate(positions_over_iters):

            # Current estimate as numpy array
            pos = np.asarray(pos)

            # Compute estimated bearing angle from array center
            direction_vec = pos - center

            # Convert direction vector to angle
            angle = np.arctan2(direction_vec[1], direction_vec[0])

            # Draw the scene with current estimate
            draw_ping_scene(
                hydrophone_positions=hydrophones_xy,
                pinger_pos=pinger_true,
                center=center,
                view_radius=VIEW_RADIUS,
                est_dir_angle=angle,
                est_dir_len=VIEW_RADIUS,
                ax=ax,
            )

            ax.set_title(
                f"Trial {trial+1}, GN iter {iter_idx+1}/{len(positions_over_iters)}"
                )
            
            # Pause to visualize each iteration
            plt.pause(0.1)   

        # Pause between trials
        plt.pause(0.5)


    plt.show()



if __name__ == "__main__":
    # run_sim_without_analysis() # Uncomment to run simple sim without localization
    run_sim_with_localization()
    
    
