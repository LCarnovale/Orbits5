import numpy as np
# import physics_functions

DELETE_FORCE_LOOP = True

def main():

    s = big_buffer()
    # s = solar_system_sample()
    return s


class SimulationError(Exception):
    def __init__(self, msg=None, particle=None, system=None, sim=None):
        s = ''
        if not msg:
            msg = 'A simulation error occured'
        if particle:
            msg += f'\nParticle: {particle}'
        if system:
            msg += f'\nSystem: {system}'
        if sim:
            msg += f'\nSimulation: {sim}'
        super().__init__(msg)

class SystemError(SimulationError):
    def __init__(self, msg=None, system=None):
        if not msg:
            msg = 'A System error has occured'
        super().__init__(msg=msg, system=system)



class System:
    def __init__(self, position, velocity=None, mass=None, radius=None, info=None):
        """
        args:
            required: position
            optional: velocity, mass, radius, info
            Give a scalar (eg. 0) in place of an array for any of the
            optional arrays to make it an array filled with that value with the
            same shape as position
        S = System(
            [N x d array of float], <-- Positions of all N particles
            [N x d array of float], <-- Velocities of all N particles
            [N size array of float], <-- Mass of all N particles
            [N size array of float], <-- Radius of all N particles
            info                    <-- Dictionary
        )
        'info' should contain extra desired information about each particle.
        The information contained can be anything, its shape will depend on
        how the information needs to be used. For example, to include charge
        as an attribute of all particles, you could have info be {'charge':[1, ...]}
        where the length of the array is equal to the number of particles.
        eg.:
            Colour of i'th particle == info['color'][i]

        Access with:
            >>> S = System( [[1, 1, 1]], info={'color':['green', ...],
                                               'charge':[-1, 1, ...]} )
            >>> S.color[0]
            'green'
            >>> S.charge[1]
            1
        """



        self._pos = position
        if np.isscalar(velocity) and velocity is not None:
            velocity = np.full(position.shape, velocity)
        self._vel = velocity
        if np.isscalar(mass) and mass is not None:
            mass = np.full(position.shape[0], mass)
        self._mass = mass
        if np.isscalar(radius) and radius is not None:
            radius = np.full(position.shape[0], radius)
        self._radius = radius

        if info == None:
            info = {}
        self._info = info
        self._active_mask = np.full(len(position), True)
        self._mask_enabled = True
    def __str__(self):
        s = f'<System: N = {self.N} (+{len(self.active) - self.N} dead), dim = {self.dim}>'
        return s

    def __getattr__(self, attr):
        """
        This can be used to safely call any field out of pos, vel, mass, radius and info
        """
        if attr in self.info:
            r = self.info[attr]
        elif attr in dir(self):
            r = self.__getattribute__(attr) # Be careful of recursive calls here!!!
        else:
            raise AttributeError(f"{attr} is not an attribute of 'System' nor is it in self.info")
        return r

    def __getitem__(self, slice):
        """
        Used to return a new system representing a
        sliced portion of the given system.
        """
        new_pos    = self.pos[slice]
        new_vel    = (None if np.any(self.vel == None) else self.vel[slice])
        new_mass   = (None if np.any(self.mass == None) else self.mass[slice])
        new_radius = (None if np.any(self.radius == None) else self.radius[slice])

        new_info = {}
        for i in self.info:
            try:
                new_info[i] = self.info[i][self.get_mask]
            except Exception:
                raise SystemError(f'Unable to slice info array in info[{i}].', self)

        return System(new_pos, new_vel, new_mass, new_radius, new_info)




    def set_active_mask(self, newVal=None):
        """
        Enable or disable the active_mask.
        If enabled, all calls to pos, vel, etc will be filtered
        by the active_mask before returning the array. ie, only
        values for active particles will be returned.
        If disabled, then the mask is ignored.

        Default action is to toggle the mask.

        If newVal provided, returns the original value,
        otherwise returns the new value.
        """
        if newVal is not None:
            self._mask_enabled = not self._mask_enabled
            return self._mask_enabled
        else:
            oldVal = self._mask_enabled
            self._mask_enabled = newVal
            return oldVal

    def add_info(self, key, array):
        """
        Adds an array to self.info under the given key
        """
        pass

    def add_particle(self, position, velocity=None, mass=None, radius=None):
        """
        Requires a value for all arguments parsed when the object was created,
        including a dict for the new info entry, (the dict that should be returned
        by info[i] for the new i), if info was provided to __init__().
        """
        def check_dim(arr_self, arr_new):
            try:
                np.alen(arr_new)
            except:
                return False

            return np.alen(arr_self[0]) == np.alen(arr_new)
        def raise_dim_err(s, given, wanted=self._pos):
            raise SystemError(msg=f"Dimension mismatch in {s} for adding particle \n"  \
                                + f"Wanted {np.alen(wanted[0])}, got {np.alen(given)}", system=self)

        def raise_arg_err(s, wanted=True):
            if wanted:
                raise SystemError(msg=f"Expected an argument for '{s}'", system=self)
            else:
                raise SystemError(msg=f"Given argument for '{s}' that was not initially given to __init__",
                                  system=self)

        if not check_dim(self._pos, position):
            raise_dim_err('position', position)

        self._pos = np.append(self._pos, [position], axis=0)
        self._active_mask = np.append(self._active_mask, True)

        attrs = ['_vel', '_mass', '_radius']
        kwargs_given = [velocity, mass, radius]
        labels = ['velocity', 'mass', 'radius']

        # set and check the other arguments:
        for attr, arg_given, label in zip(attrs, kwargs_given, labels):
            arg_self = self.__getattribute__(attr)
            if arg_given:
                if arg_self is not None:
                    if not check_dim(arg_self, arg_given):
                        raise_dim_err(label, arg_given, wanted=arg_self)

                    self.__setattr__(attr, np.append(arg_self, [arg_given], axis=0))
                else:
                    raise_arg_err(label, False)
            else:
                if arg_self is not None:
                    raise_arg_err(label, True)


    def del_particle(self, i):
        """
        Removes the i'th particle from the system.
        """
        pass

    def kill_particle(self, i):
        """
        Marks the i'th particle as dead, making it ignored by
        the simulation

        """
        self._active_mask[i] = False

    def get_collisions(self):
        """
        Returns pairs of indexes of particles colliding with eachother
        eg. -> array([ (1,3), (0, 5) ])

        calling kill_particle(np.ravel(check_collision())) will
        'kill' all colliding particles.
        calling kill_particle(check_collision()[:0 or 1]) will
        'kill' one of the colliding particles
        """
        POS_ALL = np.tile(self.pos, (self.N, 1, 1))
        POS_S = np.tile(self.pos, (1, 1, self.N)).reshape(POS_ALL.shape)
        D = POS_S - POS_ALL # r
        D = np.sqrt(np.sum(D**2, axis=-1))
        # Get combined radii:
        RM = self.radius.reshape(1, -1)
        Rgrid = RM + RM.transpose()
        collision_mask = D < Rgrid
        # Make coordinate grid:
        x = np.arange(D.shape[-1])
        y = np.arange(D.shape[-1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        # Collisions:
        row_ = xx[collision_mask]
        col_ = yy[collision_mask]
        row = row_[col_ > row_]
        col = col_[col_ > row_]
        idx_array = np.array([row, col]).transpose()
        return idx_array

    def kill_collisions(self, test, kill_func):
        """
        Requires a test function 'test', and a kill function 'kill_func':
        test(sys, A, B) -> -1|1
            Should return 1 if B is to be killed, or -1 if A is to be killed.
            The function will be given the system, and the indexes of A and B.
            *** A and B may be 1D arrays
        kill_func(sys, A, B) -> -1 or particle index
            Should make any desired modifications
            (such as changing mass, velocity, etc.) to A, as a result
            a of 'killing' B. ie, A is the survivor of the collision between
            A and B.
            *** If B is to be actually killed / inactivated, the function should return B

        Returns the number of particles 'deactivated'.
        """
        collisions = self.get_collisions()
        if not collisions.any(): return 0
        print(collisions)
        r = test(self, collisions[:,0], collisions[:,1])
        mask = np.array([r < 0, r > 0]).transpose()
        idx_array = collisions[mask]
        A = collisions[np.invert(mask)]
        B = collisions[mask]
        kill_list = np.full(len(A), -1)
        if DELETE_FORCE_LOOP:
            # A = self.active_map[A]
            # B = self.active_map[B]
            # _ = self.set_active_mask(False)
            for i, (a, b) in enumerate(zip(A, B)):
                # a = self.active_map[a]
                # b = self.active_map[b]
                kill_list[i] = kill_func(self, a, b)
            # self.set_active_mask(_)
        else:
            kill_list = kill_func(self, A, B)
        self.kill_particle(kill_list[kill_list>=0])
        return len(idx_array)

    def remove_dead(self):
        """
        Removes particles marked inactive completely
        """
        pass

    def set_pos(self, indexes, values):
        """
        Used to reliably set values in self.pos without problems
        occuring due to masks.

        Would normally be equivalent to self.pos[indexes] = values
        but that may not work sometimes due to masks.
        """
        indexes = self.active_map[indexes]
        _p = self._pos
        _p[indexes] = values
        np.copyto(self._pos, _p)

    def set_vel(self, indexes, values):
        indexes = self.active_map[indexes]
        _v = self._vel
        _v[indexes] = values
        np.copyto(self._vel, _v)
        # v = self.
        # v[indexes] = values

    def set_mass(self, indexes, values):
        # indexes = self.active_map[indexes]
        _m = self._mass
        _m[indexes] = values
        np.copyto(self._mass, _m)

    def set_radius(self, indexes, values):
        # indexes = self.active_map[indexes]
        _r = self.radius
        _r[indexes] = values
        np.copyto(self.radius, _r)

    @property
    def active_map(self):
        """
        Returns np.flatnonzero(self._active_mask)
        To allow mapping from a masked index (of an active particle)
        to a true unmasked index
        """
        return np.flatnonzero(self._active_mask)

    @property
    def active(self):
        return self._active_mask

    @property
    def dead(self):
        # Return a mask of 'dead' particle
        return np.logical_not(self._active_mask)

    @property
    def get_mask(self):
        """
        Returns the self._active_mask if self._mask_enabled is True,
        otherwise returns a self.N length array of True's.
        """
        if self._mask_enabled:
            return self._active_mask
        else:
            return np.full(self.Nfull, True)

    @property
    def N(self):
        return self._pos[self.get_mask].shape[0]

    @property
    def Nfull(self):
        # Same as N but always ignores mask
        return self._pos.shape[0]

    @property
    def dim(self):
        return self._pos.shape[1]

    def pos():
        doc = "The pos property."
        def fget(self):
            return self._pos[self.get_mask]
        def fset(self, value):
            self._pos[self.get_mask] = value
        def fdel(self):
            del self._pos
        return locals()
    pos = property(**pos())

    def vel():
        doc = "The vel property."
        def fget(self):
            if np.any(self._vel==None):
                return None
            else:
                return self._vel[self.get_mask]
        def fset(self, value):
            self._vel[self.get_mask] = value
        def fdel(self):
            del self._vel
        return locals()
    vel = property(**vel())

    def mass():
        doc = "The m property."
        def fget(self):
            if np.any(self._mass==None):
                return None
            else:
                return self._mass[self.get_mask]
        def fset(self, value):
            self._mass[self.get_mask] = value
        def fdel(self):
            del self._mass
        return locals()
    mass = property(**mass())

    def radius():
        doc = "The radius property."
        def fget(self):
            if np.any(self._radius==None):
                return None
            else:
                return self._radius[self.get_mask]
        def fset(self, value):
            self._radius[self.get_mask] = value
        def fdel(self):
            del self._radius
        return locals()
    radius = property(**radius())

    def info():
        doc = "The info property."
        def fget(self):
            return self._info
        def fset(self, value):
            self._info = value
        def fdel(self):
            del self._info
        return locals()
    info = property(**info())

def leapfrog_init(sys, f_func, t_step):
    # Go a half step backwards:
    F = f_func(sys)
    A = F / sys.mass.reshape(sys.N, 1)
    sys.vel = sys.vel - 1/2 * t_step * A

    return sys
DEFAULT_INIT = leapfrog_init

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
DEFAULT_TEST = test_mass

warning_flag = False
def kill_conserve_mass_momentum(sys, A, B):
    """
    Modifies sys where A survives a collision between A and B
    Conserves mass, momentum and centre of mass.
    """
    global warning_flag
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


    if not (sys.mass[A] == m_new).all() and not warning_flag:
        print("Warning: simultaneous collisions involving a single particle may be occuring.")
        print("These collisions may result in unexpected behaviour.")
        print("Turn on the DELETE_FORCE_LOOP option to avoid this.")
        warning_flag = True
    return B# sys.kill_particle(B)
    # return sys

DEFAULT_KILL = kill_conserve_mass_momentum

class Buffer:
    """
    Used to handle the output from sim.buffer(),
    a Buffer object can be indexed and sliced to return
    a dict with the same original keys but with specific frames.
    """
    def __init__(self, buffer_dict):
        self._dict = buffer_dict

    def __getitem__(self, s):
        if type(s) in [int, slice]:
            out = {}
            for k in self._dict:
                out[k] = self._dict[k][s]
            return Buffer(out)
        elif s in self._dict:
            return self._dict[s]
        else:
            raise KeyError

    def __len__(self):
        l = len(next(iter(self._dict.values())))
        return l

    def pull(self):
        """
        Return the first frame and delete it.
        """
        if len(self) == 0:
            return None
        f = self[0].copy()
        self = self[1:]
        return f

    def copy(self):
        return Buffer(self._dict.copy())

    @property
    def size(self):
        return len(self)

    @property
    def keys(self):
        return self._dict.keys()


class Simulation:
    def __init__(self, system, func, t_step=1, camera=None,
                test_func=DEFAULT_TEST, init_func=DEFAULT_INIT,
                kill_func=DEFAULT_KILL):
        """
        A simulation of a system
        system: System object (Must have mass)
        func:   Calculate the force on particles in system,
                from the information in system.
                func must take exactly one argument, a System object.
                func must return an array of vectors, representing force,
                where:
                    func(sys).shape == sys.pos.shape
        """
        self._sys = system
        self._func = func
        self._t_step = t_step
        self._camera = camera
        self._test_func = test_func
        self._init_func = init_func
        self._kill_func = kill_func
        self._pause = False


    def buffer(self, n, t_step=None, buffer_attrs=None):
        """
        sim.buffer(n, t_step=sim.t_step, buffer_attrs=['pos', 'vel', 'mass', 'radius'])
        --> {x: array, ... for x in buffer_attrs}

        Perform n steps and store the results in a dictionary of arrays.

        buffer_attrs should be a list of valid attributes of the system attached
        to the sim, ie sim.sys.

        The output dict has keys equal to the values in buffer_attrs,
        and each value is an array of shape (n, <the shape of sys.attr>),
        ie the array from sys.attr is stored once for each step.

        """
        if t_step == None:
            t_step = self.t_step
        if buffer_attrs == None:
            buffer_attrs = ['pos', 'vel', 'mass', 'radius', 'active']

        output = {}
        for attr in buffer_attrs:
            attr_val = self._sys.__getattr__(attr)
            output[attr] = np.zeros((n,) + np.shape(attr_val), dtype=attr_val.dtype)

        def fill_vals(i):
            for attr in buffer_attrs:
                # if attr != 'active':
                #     mask = self._sys.active
                # else:
                #     mask = slice(None) # index full array
                if attr != 'active':
                    mask = self._sys.get_mask
                else:
                    mask = slice(None)
                np.copyto(output[attr][i][mask], self._sys.__getattr__(attr))


        for i in range(n):
            # self._sys.set_active_mask(False)
            fill_vals(i)
            # self._sys.set_active_mask(True)
            self.step(t_step)
            self.step_collisions()

        return Buffer(output)

    def init_sim(self):
        self._sys = self._init_func(self._sys, self._func, self._t_step)

    def step(self, t_step = None, n=1):
        """
        If the sim is not paused,
        Calls self.func and performs a step.
        """
        if self._camera:
            self._camera.step(t_step)

        if self._pause: return

        if t_step is None:
            t_step = self._t_step

        for i in range(n):
            F = self._func(self._sys)
            A = F / self._sys.mass.reshape(self._sys.N, 1)
            self._sys.vel = self._sys.vel + t_step * A
            self._sys.pos = self._sys.pos + t_step * self._sys.vel


    def step_collisions(self):
        """
        Checks for collisions and kills particles based on the assigned test_func.
        """
        self._sys.kill_collisions(self._test_func, self._kill_func)

    def pause(self, on_off=None):
        """
        Toggle (default action) or set the pause option for the simulation.
        When paused, no force functions are called, and calling step()
        makes no changes to the system.
        """

        if on_off == None:
            self._pause = not self._pause
        else:
            self._pause = on_off


    @property
    def sys(self):
        return self._sys

    @property
    def func(self):
        return self._func

    def t_step():
        doc = "The t_step property."
        def fget(self):
            return self._t_step
        def fset(self, value):
            self._t_step = value
        return locals()
    t_step = property(**t_step())



def big_buffer(N=100, frames=100, show=False):
    try:
        gfunc
    except:
        from physics_functions import GravityNewtonian as gfunc
    c =      np.array([[np.cos(x)*10+np.random.normal(), np.sin(x)*10+np.random.normal(), 0.5*np.random.normal()] for x in np.linspace(0, 2*np.pi, N, False)])
    v = 30 * np.array([[np.sin(x)*5+np.random.normal(),-np.cos(x)*5+np.random.normal(), np.random.normal()] for x in np.linspace(0, 2*np.pi, N, False)])
    p = np.random.random((N, 3)) * 10
    # v = np.random.random((N, 3))
    r = np.ones(N) * 1e-1
    m = (1 + np.random.random(N)) * 10
    Sys = System(c, v, m, r)
    Sim = Simulation(Sys, gfunc, t_step=0.0005)
    print("Buffering...")
    d = Sim.buffer(frames)
    print("Done.")
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        active = d['active']
        for i in range(Sys.Nfull):
            p = d['pos'][:, i][active[:, i]]
            x = p[:, 0]
            y = p[:, 1]
            z = p[:, 2]
            ax.plot(x, y, z, marker='.')
        print("Plotting...")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        ax.set_zlim(-20, 20)
        plt.show()
    return d, Sim

def solar_system_sample():
    try:
        gfunc
    except:
        from physics_functions import GravityNewtonian as gfunc

    p1 = np.array([0, 0, 0], dtype=float)
    p2 = np.array([10, 0, 0], dtype=float)
    p3 = np.array([11, 0, 0], dtype=float)
    p4 = np.array([20, 0, 0], dtype=float)
    S = [0, 1, 2]
    r  = np.array([1, 0.2, 0.01, 1])[S]
    m = np.array([100, 10, 1, 0.2])[S]
    Sys = System(np.array([p1, p2, p3, p4])[S], velocity=0.0, mass=m, radius=r)
    Sim = Simulation(Sys, gfunc, t_step=0.005)

    from physics_functions import circularise
    circularise(Sys, 1, 0, Sim.func, [0, 0, 1])
    circularise(Sys, 2, 1, Sim.func, [0, 0, 1])
    print("circularised speeds:")
    print(Sys.vel)

    print("Buffering...")
    d = Sim.buffer(10000)
    print("Done.")
    active = d['active']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(Sys.Nfull):
        p = d['pos'][:, i][active[:, i]]
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        ax.plot(x, y, z, marker='.')
    print("Plotting...")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.set_zlim(-20, 20)
    plt.show()

    return locals()

if __name__ == '__main__':
    from physics_functions import GravityNewtonian as gfunc
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    s = main()
