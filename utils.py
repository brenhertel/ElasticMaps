import numpy as np

def curvature_weighting(demos):
    n_demos = len(demos)
    ind_weights = []
    for i in range(n_demos):
        traj = demos[i]
        n_pts, n_dims = np.shape(traj)
        crv_weights = np.zeros((n_pts, ))
        for j in range(1, n_pts-1):
            crv_weights[j] = np.linalg.norm(traj[j-1] - 2*traj[j] + traj[j+1])
        ind_weights.append(crv_weights)
    return np.hstack(ind_weights)
    
def jerk_weighting(demos):
    n_demos = len(demos)
    ind_weights = []
    for i in range(n_demos):
        traj = demos[i]
        n_pts, n_dims = np.shape(traj)
        jrk_weights = np.zeros((n_pts, ))
        for j in range(1, n_pts-2):
            jrk_weights[j] = np.linalg.norm(traj[j-1] - 3*traj[j] + 3*traj[j+1] - traj[j+2])
        ind_weights.append(jrk_weights)
    return np.hstack(ind_weights)