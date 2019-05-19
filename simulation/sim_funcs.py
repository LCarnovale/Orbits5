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
    F = F['force']
    A = F / sys.mass.reshape(-1, 1)
    sys.vel = sys.vel - 1/2 * t_step * A

    return sys

def leapfrog_step(sys, f_func, t_step):
    """
    Leapfrog integration:
    a_{i} = F(x_i)
    v_{i + 1/2} = v_{i-1/2} + a_{i}*dt
    x_{i + 1} = x_{i} + v_{i+1/2} * dt
    Requires
    """
    out = f_func(sys)
    a = out['force'] / sys.mass.reshape(-1, 1)
    v_mid = sys.vel + a*t_step
    x_full = sys.pos + v_mid*t_step
    sys.vel = v_mid
    sys.pos = x_full
    return out

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
    Requires sys to have mass, velocity, and radius
    """
    global _warning_flag
    # print(f'{A} killing {B}')
    pA = sys.pos[A];  pB = sys.pos[B]
    mA = sys.mass[A]; mB = sys.mass[B]
    vA = sys.vel[A];  vB = sys.vel[B]
    rA = sys.radius[A]; rB = sys.radius[B]

    # Convert 1d mass arrays into 2d Nxdim arrays
    # to allow elementwise operations with vector arrays
    mA_d = mA.reshape(-1, 1)
    # mA_d = mA_d.repeat(sys.dim, axis=-1)
    mB_d = mB.reshape(-1, 1)
    # mB_d = mB_d.repeat(sys.dim, axis=-1)

    net_p = mA_d * vA + mB_d * vB
    m_new = mA + mB
    v_new = net_p / (mA_d + mB_d)
    CoM = (pA * mA_d + pB * mB_d) / (mA_d + mB_d)

    # Get initial density:
    new_vol = 4/3 * np.pi * (rA**3 + rB**3)
    new_radius = (new_vol * 3/4 / np.pi)**(1/3)

    sys.set('pos', CoM, index=A)
    sys.set('vel', v_new, index=A)
    sys.set('mass', m_new, index=A)
    sys.set('radius', new_radius, index=A)


    if not (sys.mass[A] == m_new).all() and not _warning_flag:
        print("""
Warning: simultaneous collisions involving a single particle may be occuring.
These collisions may result in unexpected behaviour.
Turn on the DELETE_FORCE_LOOP option to avoid this. """)
        _warning_flag = True
    return B

bounce_damping = 1
friction_coeff = 5e-3
def kill_bounce(sys, A, B):
    """
    Make colliding particles 'bounce'.
    
    Tangential components are constant,
    Normal components are reflected.
    If spin is available, impulse on spin is calculated.
    
    Total output speed is reduced by bounce_damping.
    
    Centre of mass movement is conserved.
    
    Particles are never killed, so systems with this
    will not run faster over time like other functions. 
    """
    posA = sys.pos[A];  posB = sys.pos[B]
    mA = sys.mass[A]; mB = sys.mass[B]
    vA_abs = sys.vel[A];  vB_abs = sys.vel[B]
    rA = sys.radius[A]; rB = sys.radius[B]    
    # displacement vector, distance and normalized vectors:
    vAB = posA - posB
    dAB = np.linalg.norm(vAB, 2, axis=-1)
    nAB = vAB / dAB
    # momentum:
    pA_abs = mA * vA_abs; pB_abs = mB * vB_abs
    vCoM = (pA_abs + pB_abs) / (mA + mB)
    # remove centre of mass velocity
    vA = vA_abs - vCoM
    vB = vB_abs - vCoM
    # pA = mA * vA; pb = mB * vB
    
    # In centre of mass frame, velocity and momentum are
    # baaaasicallyyy the same (for this scenario) 
    # (but not really) (but they're both conserved so y'know)
    
    # break into tangent and normal components
    v_norm_A = np.dot(vA, nAB.transpose()) * nAB
    v_norm_B = np.dot(vB, nAB.transpose()) * nAB
    v_tan_A = vA - v_norm_A
    v_tan_B = vB - v_norm_B
    
    # tangential components do not change,
    # normal components have direction flipped.
    # If the particle is heading out, then don't flip:
    radial_sp_A = np.sum(v_norm_A * vAB)
    radial_sp_B = -np.sum(v_norm_B * vAB)
    v_norm_A[radial_sp_A < 0] *= -1
    v_norm_B[radial_sp_B < 0] *= -1
    

    new_vel_A = (v_norm_A * bounce_damping + v_tan_A)
    new_vel_B = (v_norm_B * bounce_damping + v_tan_B)
    new_vel_A += vCoM
    new_vel_B += vCoM

    # Try account for spin
    try:
        sA = sys.spin[A]
        sB = sys.spin[B]
    except:
        # no spin
        pass
    else:
        # Get surface velocities
        v_surf_A = v_tan_A + rA * np.cross( nAB, sA, axis=1)
        v_surf_A_n = np.linalg.norm(v_surf_A, 2, axis=-1)
        v_surf_B = v_tan_B + rB * np.cross(-nAB, sB, axis=1)
        v_surf_B_n = np.linalg.norm(v_surf_B, 2, axis=-1)

        v_norm_A_n = np.linalg.norm(v_norm_A, 2, axis=-1)
        v_norm_B_n = np.linalg.norm(v_norm_B, 2, axis=-1)

        # Get tangential impulse
        delta_v = v_surf_A - v_surf_B
        delta_v_n = np.linalg.norm(delta_v, 2, axis=-1)
        
        # These should be multiplied by mA, mB aswell, 
        # but we can ignore them for now 
        # (they get divided out later):
        # _J_coeff = friction_coeff
        J_tan_A = friction_coeff * mA * v_norm_A_n * (v_surf_A / v_surf_A_n)
        J_tan_B = friction_coeff * mB * v_norm_B_n * (v_surf_B / v_surf_B_n)
        # If the friction and/or speed of the collision is high enough, 
        # the relative surface velocities will be reduced to zero, where friction
        # also becomes zero.
        # In this case the impulse cancels the relative surface speed,
        # so it will be just mA*delta_v etc.
        J_tan_A[np.linalg.norm(J_tan_A, 2, axis=-1) > delta_v_n*mA] = mA * delta_v
        J_tan_B[np.linalg.norm(J_tan_B, 2, axis=-1) > delta_v_n*mB] = -mB * delta_v 
        # The tangential impulse also gets added to the linear momentum 
        J_net = J_tan_A - J_tan_B
        new_vel_A -= J_net / mA
        new_vel_B += J_net / mB

        dL_A = rA * np.cross( nAB, J_tan_A, axis=1)
        dL_B = rB * np.cross(-nAB, J_tan_B, axis=1)
        # dL_A = -friction_coeff * mA * rA * np.sum(vA*nAB) * np.cross(nAB, vA) / np.linalg.norm(vA, 2, axis=-1)
        # dL_B = -friction_coeff * mB * rB * np.sum(vB*nAB) * np.cross(nAB, vB) / np.linalg.norm(vB, 2, axis=-1)
        # Note these are still both per mass, ie they should have
        # mass multiplied to be physically correct. 
        # get the current values of L for A and B
        I_A = 2/5 * rA**2
        I_B = 2/5 * rB**2
        L_A = sA * I_A
        L_B = sB * I_B
        L_new_A = L_A + dL_A
        L_new_B = L_B + dL_B
        # now we can bring mass back in
        s_new_A = L_new_A / (I_A * mA)
        s_new_B = L_new_B / (I_B * mB)
        if np.any(s_new_A == np.nan):
            raise Exception

        sys.set('spin', s_new_A, index=A)
        sys.set('spin', s_new_B, index=B)
        

    
    # set positions so they just touch at the surfaces,
    # to try avoid infinite clipping and aggressive bouncing
    clipping = (rA+rB) - dAB
    # keep mA*posA + mB*posB constant, 
    # where dxA, dxB is change in position of A or B:
    # ==> dxA*mA = -dxB*mB
    # ==> dxA = -dxB*(mB/mA)
    #     dxB = -dxA*(mA/mB) 
    # And, dxA + dxB = -clipping
    # ==> dxA(1 - mA/mB) = -clipping
    # ==> dxB(mB/mA - 1) = -clipping  
    
    eq_mass_mask = mA != mB
    dxA = np.zeros(clipping.size)
    dxB = np.zeros(clipping.size)
    if np.any(eq_mass_mask):
        dxA[eq_mass_mask] = (
            clipping[eq_mass_mask] / (
                mA[eq_mass_mask]/mB[eq_mass_mask] - 1
            )
        )
        dxB[eq_mass_mask] = (
            clipping[eq_mass_mask] / (
                1 - mB[eq_mass_mask]/mA[eq_mass_mask]
            ) * -1
        )

    dxA[mA == mB] =  (clipping / 2 - 1e-2*dAB)
    dxB[mA == mB] = -(clipping / 2 - 1e-2*dAB)

    new_pos_A = posA + dxA * nAB
    new_pos_B = posB + dxB * nAB
    index_mask = clipping > 0
    sys.set('pos', new_pos_A[index_mask], index=A[index_mask])
    sys.set('pos', new_pos_B[index_mask], index=B[index_mask])

    # finally:
    sys.set('vel', new_vel_A, index=A)
    sys.set('vel', new_vel_B, index=B)

    
    return None
