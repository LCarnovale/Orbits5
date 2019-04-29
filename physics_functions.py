import numpy as np
# import sim
# System = sim.System
from sim import System

G_GRAVITATIONAL_CONSTANT = 1

def GravityNewtonian(sys):
    """
    F = -G m1 m2 / r^2

    Requires mass

    Returns Force
    """
    G = G_GRAVITATIONAL_CONSTANT

    ### Calculate a tensor for r:
    # Tile the pos array:
    POS_ALL = np.tile(sys.pos, (sys.N, 1, 1))
    POS_S = np.tile(sys.pos, (1, 1, sys.N)).reshape(POS_ALL.shape)

    D = POS_S - POS_ALL # r
    D2 = D**2 # r^2
    D2 = np.sum(D2, axis=-1) # D2 is now an N x N grid of distances (with 0 on diagonals)
    M = sys.mass.reshape(1, -1)
    M_PROD = np.matmul(M.transpose(), M)

    F_OUT = np.zeros(M_PROD.shape)
    np.divide(G * M_PROD, D2, where=D2>0, out=F_OUT)
    # F_OUT contains magnitude of forces between each particle
    F3 = F_OUT.reshape((sys.N, sys.N, 1)).repeat(sys.dim, axis=-1)
    D_norm_const = np.sqrt(D2).reshape((sys.N, sys.N, 1)).repeat(sys.dim, axis=-1) # Force times normalised direction
    D_norm = np.zeros(D.shape)
    np.divide(D, D_norm_const, where=D_norm_const>0, out=D_norm)
    F_VEC = F3 * D_norm # Now contains force vectors, F_VEC[i][j] = force between i and j
    F_NET = -np.sum(F_VEC, axis=1)
    return F_NET

if __name__ == '__main__':
    # main()
    a = np.array([[1,1], [5,1], [3, 4]])
    m = np.array([3, 6, 1])
    sys = System(a, np.zeros(a.shape), mass=m)
    print(f"Starting with positions: \n{a}\nMasses: \n{m}")
    print(f"System: {sys}")
    print("Calculating force,")
    F = GravityNewtonian(sys)
    print(f"Forces: \n{F}")
