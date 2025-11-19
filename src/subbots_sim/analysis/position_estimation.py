import numpy as np

from subbots_sim.analysis.math_tools import cylindrical_to_xy
from subbots_sim.config import global_vars
from subbots_sim.analysis.cross_correlation import compute_measured_tdoas_from_signals


def predict_TDOAs(pinger_pos, hydrophone_z, pinger_z):
    """
    Predict TDOAs given a pinger position.

    Pinger and hydrophone x,y positions are in the horizontal plane.
    All hydrophones are assumed to share the same depth hydrophone_z.

    inputs:
        pinger_pos: (2,) array-like [x_p, y_p]
        hydrophone_z: float, common z position of all hydrophones
        pinger_z: float, z position of pinger

    Returns:
        tdoas: (N-1,) array where element i-1 is Δt_i = t_0 - t_i.
    """

    # Transform hydrophone positions to cartesian coordinates
    hydrophone_pos_xy = cylindrical_to_xy(global_vars.hydrophone_positions)

    # Assuming all hydrophones are at the same z leve, calculate distances to pinger
    hydrophone_distances_to_pinger = np.sqrt((hydrophone_pos_xy[:, 0] - pinger_pos[0])**2 +
                                             (hydrophone_pos_xy[:, 1] - pinger_pos[1])**2 +
                                             (hydrophone_z - pinger_z)**2)  
    # Compute time of arrivals
    toas = hydrophone_distances_to_pinger / global_vars.speed_of_sound # t_i

    # Compute TDOAs relative to the first hydrophone
    tdoas = toas[0] - toas # deltat_i = t_0 - t_i

    # Skip hydrophone 0
    return tdoas[1:]  

def tdoa_cost_and_jacobean(pinger_pos, hydrophone_z, pinger_z, measured_tdoas):
    """
    Compute cost and jacobean for TDOA-based position estimation.

    Inputs:
        pinger_pos: (2,) array-like [x_p, y_p]
            The Candidate pinger position
        hydrophone_z: float, common z position of all hydrophones
        pinger_z: float, z position of pinger
        measured_tdoas: (N-1,) array of measured TDOAs from the cross-correlation stage
                ordered as element i-1 corresponds to hydrophone i reltaive to hydrophone 0
    Returns:
        cost: (N-1) array of residuals between predicted and measured TDOAs
            r_i = predicted_tdoa_i - measured_tdoa_i
        jacobean: (N-1, 2) array where row i-1 is the gradient of r_i with respect to pinger_pos
            J[i-1,:] = [∂r_i/∂x_p, ∂r_i/∂y_p]
    """

    # Predict TDOA's for the candidate pinger position
    predicted_tdoas = predict_TDOAs(pinger_pos, hydrophone_z, pinger_z)

    # Compute cost vector as the residuals between predicted and measured TDOAs
    cost = predicted_tdoas - measured_tdoas

    # Get hydrophone positions in cartesian coordinates
    hydrophone_pos_xy = cylindrical_to_xy(global_vars.hydrophone_positions)
    N = hydrophone_pos_xy.shape[0]

    # Set up jacobean matrix, it has one row per residual and two columns (x and y)
    jacobean = np.zeros((N-1, 2))  # (N-1, 2)
    
    # Common z difference between hydrophones and pinger
    dz = hydrophone_z - pinger_z

    # Candidate pinger position
    xp, yp = pinger_pos

    # Precompute terms for hydrophone 0
    x0, y0 = hydrophone_pos_xy[0]
    d0 = np.sqrt((x0 - xp)**2 + (y0 - yp)**2 + (dz)**2)

    # Handles case where pinger is exactly at hydrophone 0 position
    if d0 < 1e-9:
        d0_dx = 0.0
        d0_dy = 0.0
    else:
        # d0 = sqrt((xp - x0)^2 + (yp - y0)^2 + dz^2)
        # ∂d0/∂x_p = (x_p - x_0) / d0
        # ∂d0/∂y_p = (y_p - y_0) / d0
        d0_dx = (xp - x0)/d0
        d0_dy = (yp - y0)/d0

    c = global_vars.speed_of_sound

    for i in range(1, N):
        # Hydrophone i position
        xi, yi = hydrophone_pos_xy[i]

        # Distance from Hydrophone i to pinger
        di = np.sqrt((xi - xp)**2 + (yi - yp)**2 + (dz)**2)

        # Gradient of di w.r.t. pinger position
        di_dx = (xp - xi)/di
        di_dy = (yp - yi)/di

        # Δt_i = (d_0 - d_) / c  (your convention)
        # ∂Δt_i/∂x_p = (d0_dx - di_dx)/c
        # ∂Δt_i/∂y_p = (d0_dy - di_dy)/c        
        dr_dxp = (1/c) * (d0_dx - di_dx)
        dr_dyp = (1/c) * (d0_dy - di_dy)

        jacobean[i-1, 0] = dr_dxp
        jacobean[i-1, 1] = dr_dyp
    
    return cost, jacobean


def gauss_newton_step(pinger_pos, hydrophone_z, pinger_z,
                      measured_tdoas, gamma=0.0, max_step_norm=0.05):
    """
    Perform one Gauss-Newton update to improve the pinger position estimate.

    Inputs:
        pinger_pos     : (2,) array-like, current guess [x_p, y_p]
        hydrophone_z   : float, depth of all hydrophones
        pinger_z       : float, depth of the pinger
        measured_tdoas : (N-1,) array of measured TDOAs (Δt_i = t_0 - t_i)
        gamma          : float, damping term (0.0 = pure Gauss-Newton)
        max_step_norm  : float, cap on the update size (meters)

    Returns:
        new_pinger_pos : (2,) ndarray, updated [x_p, y_p]
        step           : (2,) ndarray, the actual step taken
        cost           : (N-1,) ndarray, residuals at the current position
    """

    # Compute residuals and Jacobian at the current guess
    cost, jacobean = tdoa_cost_and_jacobean(
        pinger_pos,
        hydrophone_z,
        pinger_z,
        measured_tdoas,
    )

    # Build the Gauss-Newton system (Jᵀ J + γ I) * step = -Jᵀ r
    # Where r = cost vector, J = jacobean, γ I = damping term
    JTJ = jacobean.T @ jacobean # 2x2 matrix
    JTr = jacobean.T @ cost     # 2x1 vector

    # Add the damping term to the diagonal to impove numerical stability
    H_damped = JTJ + gamma * np.eye(2)

    # Solve for the raw Gauss-Newton step
    step = -np.linalg.solve(H_damped, JTr) 

    # Clip the step size if it exceeds max_step_norm
    # This prevents overly large jumps that could destabilize convergence
    step_norm = np.linalg.norm(step)
    if (max_step_norm is not None) and (step_norm > max_step_norm):
        step = step * (max_step_norm / step_norm)

    # Update the pinger position estimate
    new_pinger_pos = pinger_pos + step

    return new_pinger_pos, step, cost



def run_gauss_newton_optimization(initial_pinger_pos, hydrophone_z, pinger_z,
                                  measured_tdoas, max_iterations=20,
                                  tol_steps=1e-6, tol_cost=1e-6,
                                  gamma=0.0, tol_grad=0.0,
                                  tol_cost_rel=0.0, max_step_norm=0.05,
                                  pinger_radius=None):

    """
    Run Gauss-Newton optimization to estimate pinger position from measured TDOAs.

    This function repeatedly calls `gauss_newton_step` starting from an
    initial guess until one of several convergence criteria is satisfied:
      * the step size ||Δr|| is small relative to the current position,
      * the gradient norm ||J^T r|| is small,
      * the relative (or absolute) change in cost (||r||^2 / 2) is small,
      * or the maximum number of iterations `max_iterations` is reached.

    Inputs:
        initial_pinger_pos : (2,) array-like initial guess [x, y]
        hydrophone_z       : float, depth of hydrophones
        pinger_z           : float, depth of pinger
        measured_tdoas     : (N-1,) array of measured Δt_i = t0 - ti
        max_iterations     : maximum number of iterations
        tol_steps          : stop when update step is very small
        tol_cost           : stop when cost value stops changing (absolute)
        tol_cost_rel       : same, but relative
        tol_grad           : stop when gradient is small (optional)
        gamma              : damping (0 = pure Gauss-Newton)
        max_step_norm      : cap on how far each step can move
        pinger_radius      : if not None, project all estimates onto a circle

    Returns:
        estimated_pinger_pos : final position estimate
        history              : dict containing positions, costs, and steps per iteration
    """

    # Srart from the initial guess
    pinger_pos = np.asarray(initial_pinger_pos, dtype=float)

    # Force the pinger position to lie on a circle of radius pinger_radius if specified
    if pinger_radius is not None:
        r = np.linalg.norm(pinger_pos)
        if r > 1e-9:
            pinger_pos = pinger_pos * (pinger_radius / r)
        else:
            # If the initial guess is exactly at the origin, nudge it
            # onto the circle along +x.
            pinger_pos = np.array([pinger_radius, 0.0], dtype=float)


    # Keeps track of optimization progress for diagnostics
    history = {
        "positions": [pinger_pos.copy()],
        "costs": [],
        "steps": [],
    }

    # Track previous cost value to check changes in cost
    previous_cost_value = None

    # Main Gauss-Newton loop
    for iteration in range(max_iterations):
        # Take one Gauss-Newton step from the current position
        #   new_pinger_pos: updated estimate after one step
        #   step:           Δr = new_pinger_pos - pinger_pos
        #   cost_vector:    residual vector r at the *current* position
        new_pinger_pos, step, cost_vector = gauss_newton_step(
            pinger_pos,
            hydrophone_z,
            pinger_z,
            measured_tdoas,
            gamma = gamma,
            max_step_norm=max_step_norm,
        )

        # Since we only care about angle, project the new estimate back
        # onto the ring of radius pinger_radius.
        if pinger_radius is not None:
            r_new = np.linalg.norm(new_pinger_pos)
            if r_new > 1e-9:
                new_pinger_pos = new_pinger_pos * (pinger_radius / r_new)
            else:
                new_pinger_pos = np.array([pinger_radius, 0.0], dtype=float)

            # Since we changed new_pinger_pos, recompute the actual step
            step = new_pinger_pos - pinger_pos


        # Compute the scalar cost value:  cost = ||r||^2 / 2
        cost_value = 0.5 * np.dot(cost_vector, cost_vector)

        # Save the diagnostic info for this iteration
        history["costs"].append(cost_value)
        history["positions"].append(new_pinger_pos.copy())
        history["steps"].append(step)

        # Stop if the step size is very small compared to the current position
        if tol_steps > 0.0:
            step_norm = np.linalg.norm(step)
            pos_norm = max(1.0, np.linalg.norm(new_pinger_pos))
            if step_norm < tol_steps * pos_norm:
                # Step is very small compared to the current position → converged
                pinger_pos = new_pinger_pos
                break

       # Stop if the gradient norm is very small 
        if tol_grad > 0.0:
            # Recompute cost and Jacobian at the new position in order to
            # evaluate the gradient norm
            _, jacobean = tdoa_cost_and_jacobean(
                new_pinger_pos,
                hydrophone_z,
                pinger_z,
                measured_tdoas,
            )
            gradient = jacobean.T @ cost_vector
            grad_norm = np.linalg.norm(gradient)

            if grad_norm < tol_grad:
                # Gradient is very small → we are near a stationary point
                pinger_pos = new_pinger_pos
                break

        # Stop if the cost value is not changing significantly (absolute or relative)
        if previous_cost_value is not None:
            abs_cost_change = abs(previous_cost_value - cost_value)
            rel_cost_change = abs_cost_change / max(1.0, previous_cost_value) 

            stop_abs = (tol_cost > 0.0) and (abs_cost_change < tol_cost)
            stop_rel = (tol_cost_rel > 0.0) and (rel_cost_change < tol_cost_rel)

            if stop_abs or stop_rel:
                pinger_pos = new_pinger_pos
                break # Converged

        # Prepare for next iteration
        pinger_pos = new_pinger_pos
        previous_cost_value = cost_value

    # Final estimated position after optimization
    estimated_pinger_pos = pinger_pos

    return estimated_pinger_pos, history




def localize_pinger_from_signals(initial_pinger_pos, hydrophone_z, pinger_z,
                                 signals, max_iterations=20, tol_steps =1e-6, tol_cost =1e-6,
                                 gamma =0.0, max_step_norm=0.05,
                                 pinger_radius=None):

    """
    Estimate the pinger's position directly from hydrophone signals.

    Steps:
        1) Convert raw signals → measured TDOAs (Δt_i = t0 - ti)
        2) Run Gauss-Newton iterations to find the best-fitting pinger position

    Inputs:
        initial_pinger_pos : (2,) array-like starting guess [x, y]
        hydrophone_z       : float, depth of all hydrophones
        pinger_z           : float, depth of the pinger
        signals            : list of N numpy arrays (one per hydrophone)
        max_iterations     : max Gauss-Newton iterations
        tol_steps          : stop when step size is tiny
        tol_cost           : stop when cost stops changing
        gamma              : damping (0 = pure Gauss-Newton)
        max_step_norm      : cap on update size
        pinger_radius      : if set, force position to lie on circle of fixed radius

    Returns:
        estimated_pinger_pos : (2,) ndarray final [x, y] estimate
        history              : dict containing positions, steps, costs per iteration
    """

    # Step 1: turn raw hydrophone signals into measured TDOAs (Δt_i = t_0 - t_i)
    measured_tdoas = compute_measured_tdoas_from_signals(signals)

    # Step 2: run Gauss-Newton to find the pinger position that best explains those TDOAs
    estimated_pinger_pos, history = run_gauss_newton_optimization(
        initial_pinger_pos=initial_pinger_pos,
        hydrophone_z=hydrophone_z,
        pinger_z=pinger_z,
        measured_tdoas=measured_tdoas,
        max_iterations=max_iterations,
        tol_steps=tol_steps,
        tol_cost=tol_cost,
        gamma=gamma,
        max_step_norm=max_step_norm,
        pinger_radius=pinger_radius,
    )

    
    # Step 3: Return estimated position and diagnostics
    return estimated_pinger_pos, history

