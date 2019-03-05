import torch

NOISE = 0.0
STRETCH = 1
TEST_DISPLACEMENT = 0

def create(N, D):
    X = torch.rand(N, D)
    Y = create_output(X)
    return add_noise(X), add_noise(Y)

def test(N, D):
    X = torch.rand(N, D) + TEST_DISPLACEMENT
    Y = create_output(X)
    return X, Y # no perturbation - we want exact answers.

# === PRIVATE ===

def create_output(X):
    # Try to model d = t*v0 + 0.5*a*t^2
    X = X * STRETCH
    t = X[:,0]
    v = X[:,1]
    a = X[:,2]
    return t*v + 0.5*a*t**2

def add_noise(M):
    return M + (torch.rand(M.size())*2-1)*NOISE
    
