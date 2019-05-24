import math
import numpy as np
import sim_funcs

default_masked_attributes = ['pos', 'vel', 'mass', 'radius', 'info']

import buffer as bu
import system as sy
# System = system.ClassicSystem 
# Buffer = buffer.Buffer

DEFAULT_BUFFER_ATTRS = ['pos', 'vel', 'mass', 'radius', 'active', 'N']

# buffer.default_masked_attributes = default_masked_attributes

def main():
    # s = big_buffer()
    # s = solar_system_sample()
    m = disc_merger()
    m.new_key('force', 3)
    s = Simulation(m, gfunc, 0.01)
    s.track("force")
    s.step(10)
    s.buffer(100)

    return locals()


class SimulationError(Exception):
    def __init__(self, msg=None, particle=None, system=None, sim=None):
        if not msg:
            msg = 'A simulation error occured'
        if particle:
            msg += f'\nParticle: {particle}'
        if system:
            msg += f'\nSystem: {system}'
        if sim:
            msg += f'\nSimulation: {sim}'
        super().__init__(msg)



DEFAULT_INIT = sim_funcs.leapfrog_init
DEFAULT_STEP = sim_funcs.leapfrog_step
DEFAULT_TEST = sim_funcs.test_mass
DEFAULT_KILL = sim_funcs.kill_conserve_mass_momentum



class Simulation:
    def __init__(self, system, func, t_step=1, camera=None,
                 test_func=DEFAULT_TEST, init_func=DEFAULT_INIT,
                 step_func=DEFAULT_STEP, kill_func=DEFAULT_KILL):
        """
        A simulation of a system
        system: System object (Must have mass)
        func:
            Calculate the force on particles in system,
            from the information in system.
            func must take exactly one argument, a System object.
            func must return an array of vectors, representing force,
            where:
                func(sys).shape == sys.pos.shape
        test_func, kill_func:
            Functions to determine the outcome of collisions, as outlined
            in the System class.
        init_func, step_func:
            Both must take (sys:<System>, f_func:(same as func above), t_step)
            and make changes to system to set up an integration method
            or perform a step.
            ** The init func must be called manually, ideally before the first
               .step() call
        """
        self._sys = system
        self._func = func
        self._t_step = t_step
        self._camera = camera
        self._test_func = test_func
        self._init_func = init_func
        self._kill_func = kill_func
        self._step_func = step_func
        self._pause = False
        self._buffer = None
        self._tracked = []
        self._last_frame = None

    def __getattr__(self, attr):
        """
        If a buffer exists, try getting the attribute from the buffer.
        Otherwise, try getting the attribute from the attached System.
        """
        if self._buffer and attr in self._buffer:
            if self._last_frame:
                out = self._last_frame[attr][0]
                active = self._last_frame.active[0]
            else:
                out = self._buffer[attr][0]
                active = self._buffer.active[0]
            if attr in self._sys.names:
                out = out[active]
            return out
        elif self._buffer and attr not in self._buffer:
            raise SimulationError(f"""
The simulation has a buffer stored, but the
requested attribute {attr} does not exist in the stored buffer.
To get reliable results, you should check if it exists in and then 
retrieve the attribute directly from the system (sim.sys)
or the buffer (sim.stored)""")
        else:
            return self._sys.get(attr)


    def buffer(self, buffer_n, t_step=None, buffer_attrs=None, append_buffer=True,
        verb=False, first_frame=True, **step_kwargs):
        """
        sim.buffer(buffer_n, t_step=sim.t_step, buffer_attrs=[],
            **step_kwargs ( used by Sim.step() ) )
        --> {x: array, ... for x in buffer_attrs}

        Perform buffer_n steps and store the results in a dictionary of arrays.
        buffer_attrs default: ['pos', 'vel', 'mass', 'radius', 'N']
        buffer_attrs should be a list of valid attributes of the system attached
        to the sim, ie sim.sys. If buffer_attrs contains '+', then the supplied attrs
        will be recorded along with the default ones.

        if append_buffer is True, any existing buffer will be appended. If the
        existing buffer has different fields recorded, an error is raised.

        Prints progress if verb == True

        if first_frame is true, the current state of the system is recorded as
        the first frame, otherwise not.

        The output dict has keys equal to the values in buffer_attrs,
        and each value is an array of shape (n, <the shape of sys.attr>),
        ie the array from sys.attr is stored once for each step.

        """
        if t_step == None:
            t_step = self.t_step
        if buffer_attrs == None:
            buffer_attrs = DEFAULT_BUFFER_ATTRS + self._tracked
        elif "+" in buffer_attrs:
            buffer_attrs.remove('+')
            buffer_attrs += DEFAULT_BUFFER_ATTRS + self._tracked
        if append_buffer and self._buffer != None and {*buffer_attrs} != set(self._buffer.keys):
            raise SimulationError(
                msg="Keys in existing do not match requested buffer_attributes, \ncannot append to the existing buffer."
            )
        if 'pull_buffer' in step_kwargs:
            raise SimulationError(
                msg="Can not use 'pull_buffer' keyword when creating a buffer. (It must be false when performing steps)"
            )

        output = {}
        for attr in buffer_attrs:
            attr_val = self._sys.get(attr, masked=False)
            try:
                dtype_ = attr_val.dtype
            except:
                dtype_ = type(attr_val)
            output[attr] = np.zeros((buffer_n,) + np.shape(attr_val), dtype=dtype_)

        def fill_vals(i):
            for attr in buffer_attrs:
                output[attr][i] = self._sys.get(attr, masked=False)



        if verb:
            from sys import stdout
            flush = stdout.flush
            steps_len = len(str(buffer_n))
        if first_frame:
            fill_vals(0)
        for i in range(1*first_frame, buffer_n):
            if verb:
                print(f'Buffering frame {i:0>{steps_len}} / {buffer_n}', end = '\r'); flush()
            self.step(t_step, pull_buffer=False, ignore_pause=True, **step_kwargs)
            fill_vals(i)
        if verb: print()

        new_buffer = bu.Buffer(output)
        if append_buffer:
            self._buffer = self._buffer + new_buffer
        return new_buffer

    def track(self, *args):
        """
        Supply with attribute names and the system will have these
        values updated in system when they are found in a step.

        The default force function returns just a value for 'force'.
        Calling sim.track('force') will have sys.force set to this
        value every step. User defined functions should return a dict
        with labels and values to represent which values are to be tracked.
        """
        if args:
            self._tracked = [*args]



    def init_sim(self):
        self._sys = self._init_func(self._sys, self._func, self._t_step)

    def step(self, t_step = None, n=1, collisions=True, mode='every', set_delta=True,
            pull_buffer=True, buffer_in_pause=False, ignore_pause=False):
        """
        If the sim is not paused,
        Calls self.func and performs a step.

        n:  Number of steps to execute. Default 1.

        collisions: Default True, If True, call step_collisions() after each step (if mode == 'every')
        or after the last step (if mode == 'once')

        mode: Default 'every', makes collision checking happen every step. If set to 'once',
            collisions are checked after the final step.

        set_delta: If True, call sys.set_pos_delta() after the last step.
            Will only be called once, not per step.

        pull_buffer: If True, a frame from the attached buffer will be pulled if it exists.

        """
        if mode not in ['once', 'every']:
            raise SimulationError(f"'mode' argument must be 'once' or 'every', given {mode}")
        if self._camera:
            self._camera.step(t_step)


        if t_step is None:
            t_step = self._t_step


        if self._pause and not ignore_pause:
            if buffer_in_pause:
                self.buffer(n, append_buffer=True, first_frame=False)
            return

        if pull_buffer:
            if self._buffer != None:
                if len(self._buffer) > 0:
                    frame = self._buffer.pull()
                    self._last_frame = frame
                    return frame
                else:
                    self._last_frame = None
                    self._buffer = None

        out = None # Initialize for later incase n==0 for some reason
        for _ in range(n):
            out = self._step_func(self._sys, self._func, self._t_step)
            # Get the current mask so that after checking collisions
            # we can use it to set the tracked values, otherwise the
            # output will be a different shape to what the new mask will need.
            out_mask = self._sys.active.copy()
                    
            if collisions and mode == 'every':
                self.step_collisions()

        if self._tracked and out:
            for x in self._tracked:
                if x in out:
                    val = out[x]
                    self._sys.set(x, val, index=np.flatnonzero(out_mask), masked=False)

        if collisions and mode == 'once':
            self.step_collisions()
            
        return None



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

    """
    Properties
    """
    @property
    def tracked(self):
        return self._tracked.copy()

    @property
    def paused(self):
        return self._pause

    @property
    def stored(self):
        return self._buffer
    
    @property
    def prepped(self):
        if self._last_frame != None:
            return self._last_frame
        else:
            return self._sys

    @property
    def sys(self):
        return self._sys

    @property
    def func(self):
        return self._func

    def kill_func():
        def fget(self):
            return self._kill_func
        def fset(self, func):
            self._kill_func = func
        return locals()
    kill_func = property(**kill_func())

    def t_step():
        doc = "The t_step property."
        def fget(self):
            return self._t_step
        def fset(self, value):
            self._t_step = value
        return locals()
    t_step = property(**t_step())

########################################################################
################## Some example systems/simulations ####################
########################################################################

def disc_sys(N, mass_props=(1, 0, 'normal'), radius_props=(10, 0, 'uniform'), 
             vel=(1, 1), thickness=0.1, particle_rad=(1, 0, 'normal'), axis=None):
    """ 
    Create a system of a disc of particles
    mass_props
    radius_props
    particle_rad :  all are of the form:
                    (mean, std-dev, 'normal') or
                    (min, max, 'uniform')
    For the mass of particles, radius of disc,
    and radius of particles respectively.


    vel: Set to ('circ', force_f) for all particles to be circularised about 
         enclosed centre of mass, otherwise:
         in the middle of the disc (r / 2), the speed is vel[0].
         and other distances are scaled by vel[1], ie:
         v(r) = ((r - r_mid)/(r_mid) * vel[1] + 1) * vel[0]
         so that if vel[1] == 0, all particles have the same speed,
         and if vel[1] == 1, speed at r = 0 is 0, and at max radius
         is 2*vel[0] (ie all particles have the same angular speed).
         All velocities are tangential.

    axis: Default [0, 0, 1], axis of rotation and normal to the disc.
    
    thickness:
        the position along axis is distributed normally with a
        standard deviation of 'thickness'
    """
    from physics_functions import circularise
    if np.any(axis == None):
        axis = [0, 0, 1]

    def rand_f(a, b, type_, size):
        if type_ == 'normal':
            return np.random.normal(loc=a, scale=b, size=size)
        elif type_ == 'uniform':
            return np.random.random(size=size) * (b - a) + a
        else:
            raise Exception

    r = rand_f(*radius_props, N)            # radius (disc)
    m = rand_f(*mass_props, N)              # mass
    r_mid = np.mean(r)         
    r_p = rand_f(*particle_rad, N)          # radius (particles)
    z_unit = np.asarray(axis) / math.sqrt(np.dot(axis, axis))
    temp_vec = 1. * np.asarray(axis)
    temp_vec[0] += 1 
    temp_vec[1] -= 1
    x_unit = np.cross(temp_vec, z_unit)
    x_unit /= math.sqrt(np.dot(x_unit, x_unit))
    y_unit = np.cross(z_unit, x_unit)
    y_unit /= math.sqrt(np.dot(y_unit, y_unit))
    angle = np.random.random(N) * 2*np.pi   # angle in disc
    x  = r * np.cos(angle);# x = x.transpose()
    y  = r * np.sin(angle);# y = y.transpose()  
    z = rand_f(0, thickness, 'normal', N)   # position along (local) z-axis
    pos = z_unit * z.reshape(-1, 1) + x_unit * x.reshape(-1, 1) + y_unit * y.reshape(-1, 1)
    if vel[0] == 'circ':
        # sort = np.argsort(r)
        # m_enc = np.array([
        #     np.sum(m[sort[:i]])
        # ])
        # assume enclosed mass is r^2/R^2 * sum(mass)
        m_enc = r**2/max(r)**2 * np.sum(m)
        
        # circularised speed: v = sqrt(F r / m)
        temp_sys = sy.ClassicSystem(pos=pos, vel=0., mass=m)
        temp_sys_enc = sy.ClassicSystem(pos=np.array(N*[[0., 0., 0.]]), vel=0., mass=m_enc)
        temp_sys.append(temp_sys_enc)
        s = np.zeros(N)
        for i in range(N):
            circularise(temp_sys, i, i+N, vel[1], z_unit)
            s[i] = np.linalg.norm(temp_sys.vel[i], 2)
    else:
        s = ((r - r_mid)/r_mid * vel[1] + 1) * vel[0] # speed
    vx = s * np.sin(angle);# vx = vx.transpose() 
    vy = s * -np.cos(angle);# vy = vy.transpose() 
    vel = x_unit * vx.reshape(-1, 1) + y_unit * vy.reshape(-1, 1)
    sys = sy.ClassicSystem(pos=pos, vel=vel, mass=m, radius=r_p)
    
    return sys

def big_buffer(N=100, frames=100, show=False, **disc_kwargs):
    # Set show = True to plot on a 3D plot
    try:
        gfunc
    except:
        from physics_functions import GravityNewtonian as gfunc

    Sys = disc_sys(N, **disc_kwargs)
    Sim = Simulation(Sys, gfunc, t_step=0.0005)
    print("Buffering...")
    d = Sim.buffer(frames, verb=True)
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

def disc_merger(N=100, disc_radii=8, kick=[0, 0, 0], axis2=[0, 0, 1], **disc_kwargs):
    kwargs = {
        'mass_props':(10, 5, 'normal'),
        'radius_props':(disc_radii, 0, 'uniform'),
        'particle_rad':(0.05, 0, 'normal'),
        'vel':(100, 1)
    }
    kwargs.update(disc_kwargs)
    disc1 = disc_sys(N, **kwargs)
    kwargs['axis'] = axis2
    disc2 = disc_sys(N, **kwargs)
    shift = np.array([[0., 4*disc_radii, 0.]])
    disc1.pos += shift
    disc1.vel += np.asarray(kick)
    disc2.pos -= shift
    disc1.append(disc2)
    return disc1

def small_galaxy(N=1000):
    try:
        gfunc
    except:
        from physics_functions import GravityNewtonian as gfunc
    p = np.random.normal(scale=(10., 10., .2), size=(N, 3))
    r = np.linalg.norm(p, 2, axis=-1)
    a = 30; b = 1 # Constants, can be changed to try different rotation curve
    speed = 100*a/(a*r + b) * np.log(a*r + b)
    v = speed.reshape(-1, 1) * np.cross([0.,0.,.1], p)/r.reshape(-1, 1)
    m = np.full(len(p), 0.01)
    r = np.full(len(p), 0.01)
    Sys = sy.ClassicSystem(pos=p, vel=v, mass=m, radius=r)
    Sim = Simulation(Sys, gfunc, t_step=0.005)
    # print(Sys.mass.shape)
    # print(Sys.N)
    # print(Sys.dim)
    return Sim, Sys

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
    Sys = sy.ClassicSystem(pos=np.array([p1, p2, p3, p4])[S], vel=0.0, mass=m, radius=r)
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
