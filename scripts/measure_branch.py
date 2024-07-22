import numpy as np

def distance(branch):
    return np.float64(np.sum(np.linalg.norm(branch[1:]-branch[:-1], axis=1)))

def angle(x,y,z):
    u = y-x
    v = z-x
    return np.degrees(np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))

def angle_metric(lb, nlb, rb, nrb):
    a = angle(lb[0], lb[-1], rb[-1])
    b = angle(nlb[0], nlb[-1], nrb[-1])
    return abs(a-b)

def branches_metric(branch, new_branch):
    PD = abs(np.sum(np.linalg.norm(branch[1:]-branch[:-1], axis=1)) - np.sum(np.linalg.norm(new_branch[1:]-new_branch[:-1], axis=1)))
    SP = np.sum(np.linalg.norm(branch-new_branch, axis=1))
    N = len(branch)
    DTW = float("inf") * np.ones((N, N))
    DTW[0, 0] = 0
    for i in range(1, N):
        for j in range(1, N):
            cost = np.linalg.norm(branch[i] - new_branch[j])
            DTW[i, j] = cost + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
    return PD, SP, DTW[N-1, N-1]