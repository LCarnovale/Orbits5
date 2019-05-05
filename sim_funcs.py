import numpy as np

def RK4_init(sys, f_func, t_step):
    """
    Fourth order Runge Kutta integration,
    described in RK4_step() function.

    Nothing needed to init.
    """
    return


def RK4_step(sys, f_func, t_step):
    """
    Runge-Kutta (4) integration:
    if:
        d/dt(y) = f(t, y), y(t0) = y0
    Then:
        y{n + 1} = y{n} + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        t{n + 1} = t{n} + h
    where:
        k1 = h*f1 = h*f(tn, yn)
        k2 = h*f2 = h*f(tn + h/2, yn + k1/2)
        k3 = h*f3 = h*f(tn + h/2, yn + k2/2)
        k4 = h*f4 = h*f(tn + h, yn + k3)

    f(t, y) (ie the force function) gives our acceleration,
    hence y is our velocity.
    *** In this function, the force function
        will be assumed to have no time dependence, although the position
        will be slightly varied throughout the integration

    We are given t_step, and will let this be our 'h'.
    """
    h = t_step
    # note that f1 is an acceleration, not a force
    temp_sys = sys[:]
    mass_ = sys.mass.reshape(-1, 1)
    f1 = f_func(temp_sys) / mass_
    k1 = h * f1
    temp_sys.pos += h / 2 * temp_sys.vel
    temp_sys.vel += k1 / 2
    k2 = h * f_func(temp_sys) / mass_
    temp_sys.vel += k2 / 2
    k3 = h * f_func(temp_sys) / mass_
    temp_sys.pos += h / 2 * temp_sys.vel
    temp_sys.vel += k3
    k4 = h * f_func(temp_sys)

    new_vel =  1./6*(k1 + 2*k2 + 2*k3 + k4)
    sys.pos += h*(sys.vel)# + new_vel)
    sys.vel += new_vel
    return sys



def leapfrog_init(sys, f_func, t_step):
    # Go a half step backwards:
    F = f_func(sys)
    A = F / sys.mass.reshape(sys.N, 1)
    sys.vel = sys.vel - 1/2 * t_step * A

    return sys

def leapfrog_step(sys, f_func, t_step):
    """
    Leapfrog integration:
    a_{i} = F(x_i)
    v_{i + 1/2} = v_{i-1/2} + a_{i}*dt
    x_{i + 1} = x_{i} + v_{i+1/2} * dt
    """
    a = f_func(sys) / sys.mass.reshape(-1, 1)
    v_mid = sys.vel + a*t_step
    x_full = sys.pos + v_mid*t_step
    sys.vel = v_mid
    sys.pos = x_full
    return sys

def test_mass(sys, A, B):
    """
    Favours a more massive particle in collisions
    Returns 1 if B is to be killed,
    else -1 if A is.
    """
    massA = sys.mass[A]
    massB = sys.mass[B]
    r = np.zeros(np.shape(A), dtype=int)
    r[massA >  massB] = 1
    r[massA <= massB] = -1
    return r

_warning_flag = False
def kill_conserve_mass_momentum(sys, A, B):
    """
    Modifies sys where A survives a collision between A and B
    Conserves mass, momentum and centre of mass.
    """
    global _warning_flag
    # print(f'{A} killing {B}')
    pA = sys.pos[A];  pB = sys.pos[B]
    mA = sys.mass[A]; mB = sys.mass[B]
    vA = sys.vel[A];  vB = sys.vel[B]

    # Convert 1d mass arrays into 2d Nxdim arrays
    # to allow elementwise operations with vector arrays
    mA_d = mA.reshape(-1, 1)
    mA_d = mA_d.repeat(sys.dim, axis=-1)
    mB_d = mB.reshape(-1, 1)
    mB_d = mB_d.repeat(sys.dim, axis=-1)

    net_p = mA_d * vA + mB_d * vB
    m_new = mA + mB
    v_new = net_p / (mA_d + mB_d)
    CoM = (pA * mA_d + pB * mB_d) / (mA_d + mB_d)

    # Get initial density:
    init_density_inverse = (4/3 * np.pi * sys.radius[A]**3) / mA
    new_radius = init_density_inverse * m_new
    new_radius = (new_radius * 3/4 / np.pi)**(1/3)

    sys.set_pos(A, CoM)
    sys.set_vel(A, v_new)
    sys.set_mass(A, m_new)
    sys.set_radius(A, new_radius)


    if not (sys.mass[A] == m_new).all() and not _warning_flag:
        print("""
Warning: simultaneous collisions involving a single particle may be occuring.
These collisions may result in unexpected behaviour.
Turn on the DELETE_FORCE_LOOP option to avoid this. """)
        _warning_flag = True
    return B
