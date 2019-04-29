import numpy as np
# import physics_functions

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
        'info' should contain relevant information about each particle.
        The information contained can be anything, but should be
        consistant among particles (but most functions will work if not).
        The keys in info should correspond to
        the index of the particle in the position and velocity arrays.

        eg.:
            Colour of i'th particle == info[i]['color']

        Access with:
            >>> S = System( [[1, 1, 1]], info={0:{'color':'green'}} )
            >>> S.color[0]
            'green'
        """

        self._pos = position
        if velocity is not None and np.isscalar(velocity):
            velocity = np.ones(position.shape) * velocity
        self._vel = velocity
        if mass is not None and np.isscalar(mass):
            mass = np.ones(position.shape[0]) * mass
        self._mass = mass
        if radius is not None and np.isscalar(radius):
            radius = np.ones(position.shape[0]) * radius
        self._radius = radius

        self._info = info
        self._active_mask = np.full(len(position), True)

    def __str__(self):
        s = f'<System: N = {self.N}, dim = {self.dim}>'
        return s

    def add_particle(self, position, velocity=None, mass=None, radius=None, info=None):
        """
        Requires a value for all arguments parsed when the object was created,
        including a dict for the new info entry, (the dict that should be returned
        by info[i] for the new i), if info was provided to __init__().
        """
        check_dim = lambda arr_self, arr_new: arr_self.shape[-1] == arr_new.shape[0]
        def raise_dim_err(s, given):
            raise SystemError(msg=f"Dimension mismatch in {s} for adding particle \n"  \
                                + f"Wanted {self._pos.shape}, got {given.shape}", system=self)

        def raise_arg_err(s, wanted=True):
            if wanted:
                raise SystemError(msg=f"Expected an argument for '{s}'", system=self)
            else:
                raise SystemError(msg=f"Given argument for '{s}' that was not initially given to __init__",
                                  system=self)

        if not check_dim(self._pos, position):
            raise_dim_err('position', position)

        self._pos = np.append(self._pos, position)

        attrs = ['_vel', '_mass', '_radius']
        kwargs_given = [velocity, mass, radius]
        labels = ['velocity', 'mass', 'radius']

        # set and check the other arguments:
        for attr, arg_given, label in zip(attrs, kwargs_given, labels):
            arg_self = self.__getattribute__(attr)
            if arg_given:
                if arg_self is not None:
                    if not check_dim(arg_self, arg_given):
                        raise_dim_err(label, arg_given)

                    self.__setattr__(attr, np.append(arg_self, arg_given))
                else:
                    raise_arg_err(label, False)
            else:
                if arg_self is not None:
                    raise_arg_err(label, True)

        # check info:
        if info:
            if self._info is not None:
                # N is now the length of the new array
                self._info[self.N - 1] = info.copy()
            else:
                raise_arg_err('info', False)
        else:
            if self._info is not None:
                raise_arg_err('info', True)

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

    def check_collision(self):
        """
        Returns pairs of indexes of particles colliding with eachother
        eg. -> array([ (1,3), (0, 5) ])

        calling kill_particle(np.ravel(check_collision())) will
        'kill' all colliding particles.
        calling kill_particle(check_collision()[:0 or 1]) will
        'kill' one of the colliding particles
        """
        POS_ALL = np.tile(self.pos, (self.N, 1, 1))
        POS_S = np.tile(self.pos, (1, 1, sys.N)).reshape(POS_ALL.shape)
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






    @property
    def active(self):
        return self._active_mask

    @property
    def dead(self):
        # Return a mask of 'dead' particle
        return np.logical_not(self._active_mask)

    @property
    def N(self):
        return self._pos[self._active_mask].shape[0]

    @property
    def dim(self):
        return self._pos.shape[1]

    def pos():
        doc = "The pos property."
        def fget(self):
            return self._pos[self._active_mask]
        def fset(self, value):
            self._pos[self._active_mask] = value
        def fdel(self):
            del self._pos
        return locals()
    pos = property(**pos())

    def vel():
        doc = "The vel property."
        def fget(self):
            return self._vel[self._active_mask]
        def fset(self, value):
            self._vel[self._active_mask] = value
        def fdel(self):
            del self._vel
        return locals()
    vel = property(**vel())

    def mass():
        doc = "The m property."
        def fget(self):
            return self._mass[self._active_mask]
        def fset(self, value):
            self._mass[self._active_mask] = value
        def fdel(self):
            del self._mass
        return locals()
    mass = property(**mass())

    def radius():
        doc = "The radius property."
        def fget(self):
            return self._radius[self._active_mask]
        def fset(self, value):
            self._radius[self._active_mask] = value
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

class Simulation:
    def __init__(self, system, func, t_step=1):
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

    def init_sim(self):
        self._sys = leapfrog_init(self._sys, self._func, self._t_step)

    def step(self, t_step = None, n=1):
        """
        Calls self.func and performs a step.
        """
        if t_step is None:
            t_step = self._t_step

        for i in range(n):
            F = self._func(self._sys)
            A = F / sys.mass.reshape(sys.N, 1)
            sys.vel = sys.vel + t_step * A
            sys.pos = sys.pos + t_step * sys.vel

    @property
    def sys(self):
        return self._sys

if __name__ == '__main__':
    from physics_functions import GravityNewtonian as gfunc
    # a = np.array([[1,1],[5,1],[3,4]])
    # m = np.array([1, 6, 4])
    # sys = System(a, velocity=0, mass=m)
    # sim = Simulation(sys, gfunc, 0.1)
    # sim.step()

    import matplotlib.pyplot as plt
    Nparticles = 10
    c = np.array([[np.cos(x), np.sin(x)] for x in np.linspace(0, 2*np.pi, Nparticles, False)]) * 5
    v = np.array([[np.sin(x), -np.cos(x)] for x in np.linspace(0, 2*np.pi, Nparticles, False)]) * 1
    # m = np.ones(len(c)) * 0.2
    m = np.random.random(len(c)) * 0.02
    r = np.random.random(len(c))
    sys = System(c, velocity=v, radius=r, mass=m)
    sim = Simulation(sys, gfunc, 0.01)
    sim.init_sim()
    iters = 100
    pos_array = np.zeros((iters,) + c.shape)
    print("Starting")
    # for i in range(iters):
    #     sim.step()
    #
    #     np.copyto(pos_array[i], sim.sys.pos)
    print("Done")

    for p in range(len(c)):
        x = pos_array[:,p,0]
        y = pos_array[:,p,1]
        plt.plot(x, y, '.-')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.show()
