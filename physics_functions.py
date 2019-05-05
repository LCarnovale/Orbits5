# A list of force functions for use by sim.py
# All force functions should take exactly 1 argument, 'sys'
# They should return an array of force vectors for all
# particles in sys.

import numpy as np
from sim import System

class FunctionError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

TRACKED_VALUE = None
# def set_tracked(value):


GRAVITATIONAL_CONSTANT = 1

def circularise(sys, A, B, f_func, axis):
    """
    A, B must be scalars!
    Requires mass
    Generic circularising function
    Sets A to be in a circular orbit around B, by setting
    an angular velocity for A aligned with 'axis' to match
    an attractive force with centripetal force.

    Conserves momentum of CoM
    """
    if np.any(sys.mass==None):
        raise FunctionError("Particles in 'sys' must have mass")
    if not (np.isscalar(A) and np.isscalar(B)):
        raise FunctionError("A and B must be scalars")
    new_sys = sys[[A, B]] # Isolate the two particles
    # if np.any(new_sys.mass) and np.any(new_sys.vel):
    # print("=== new sys ===")
    # print(new_sys.mass)
    # print(new_sys.pos)
    # print('=== ===')
    pA, pB = new_sys.pos[0], new_sys.pos[1]
    BA_vec = pA - pB
    # dist = np.linalg.norm(BA_vec, 2)
    mA = new_sys.mass[0]; mB = new_sys.mass[1]
    vA = new_sys.vel[0];  vB = new_sys.vel[1]
    rMass = mA*mB / (mA + mB)
    CoM = (mA*pA + mB*pB)/(mA + mB)
    rA = pA - CoM
    rB = pB - CoM
    dA = np.linalg.norm(rA, 2)
    dB = np.linalg.norm(rB, 2)


    F = f_func(new_sys)
    F = np.linalg.norm(F, 2, axis=-1)
    fA = F[0] #/ rMass
    fB = F[1] #/ rMass

    vA_new = np.sqrt(fA * dA  / (rMass))
    vB_new = np.sqrt(fB * dB  / (rMass))

    direction  = np.cross(axis, BA_vec)
    direction /= np.linalg.norm(direction, 2)
    vA_new =  vA_new * direction
    vB_new = -vB_new * direction

    vCoM = (mA*vA + mB*vB) / (mA + mB)
    vA_new += vCoM
    vB_new += vCoM
    sys.set_vel([A,B], [vA_new, vB_new])


def GravityNewtonian(sys):
    """
    F = -G m1 m2 / r^2

    Requires mass

    Returns Force
    """
    G = GRAVITATIONAL_CONSTANT

    ### Calculate a tensor for r:
    # Tile the pos array:
    print("sys.pos:", sys.pos, "N:", sys.N)
    POS_ALL = np.tile(sys.pos, [sys.N, 1, 1])
    POS_S = np.tile(sys.pos, (1, 1, sys.N)).reshape(POS_ALL.shape)
    print("sys.pos:", sys.pos, "N:", sys.N)

    D = POS_S - POS_ALL # r vector
    D2 = np.linalg.norm(D, 2, axis=-1) # D2 is now an N x N grid of distances (with 0 on diagonals)
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
