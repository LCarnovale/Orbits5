# Perform time analysis of functions.

# import time
from time import perf_counter as time_f
import timeit
import numpy as np
from sys import stdout
flush = stdout.flush


import simulation.sim as sim
import simulation.sim_funcs as sim_f
import simulation.system as sys
import simulation.physics_functions as phys_f


print("Performance analysis of Orbits5 functions:")
print("=======================")


SYS_SIZE = [10, 50, 100, 500, 1000]
SAMPLES = 999    # Run this many trials per system size
TIME_LIMIT = 60  # Max time to spend running trials

f_func    = phys_f.GravityNewtonian
init_func = sim_f.leapfrog_init
step_func = sim_f.leapfrog_step
system    = sys.ClassicSystem
test_func = sim_f.test_mass


# Give fields and (length, scale)
fields = {
    'pos':(3, 100),
    'vel':(3, 1),
    'mass':(1, 1),
    'radius':(1, 1),
}

def null_f(*args, **kwargs):
    # A blank dummy function
    return None

def sys_init_f(N):
    # Initialise values for a system of N particles
    # return a system
    out = {}
    for f in fields:
        out[f] = (np.random.random((N, fields[f][0])) - .5)*2 * fields[f][1]
    
    return system(**out)

def timer(func, args=None, kwargs=None, reps=SAMPLES, time_lim=TIME_LIMIT):
    # Time a function with args and kwargs.
    # function is called reps times,
    # if the total time elapsed is greater than time_lim
    # then the test is stopped.
    # 
    # Returns a list of the time taken for each call.  

    if args == None: args = []
    if kwargs == None: kwargs = {}

    results = []
    test_start = time_f()
    for r in range(reps):
        r_start = time_f()
        _ = func(*args, **kwargs)
        t = time_f() - r_start

        results.append(t)
        if time_f() - test_start > time_lim:
            break
    
    return np.array(results)

title_width = 30
time_width = 22

def format_title(s):
    return f"{s:<{title_width}}"

def format_time(t_vals):
    t = np.mean(t_vals) * 1e3
    s = np.std(t_vals)  * 1e3
    c = len(t_vals)
    return f"{f'{t:.2f} ({s:.2f}) [{c:>3}]':>{time_width}}"

print("<time> (<std dev>) [<number of calls>].  Times in ms.")
print(format_title('System size:'), *[f"{x:>{time_width}}" for x in SYS_SIZE], sep='')

# Build systems
systems = []
print(format_title('Building systems:'), end="")
try:
    for n in SYS_SIZE:
        t_vals = timer(sys_init_f, args=(n,) )
        print(format_time(t_vals), end="")    
        systems.append(sys_init_f(n))
        flush()
except KeyboardInterrupt:
    print("Stopping test.")
else:
    print()

# Initialise
print(format_title("Initialising simulations:"), end="")
try:
    for s in systems:
        t_vals = timer(init_func, args=(s, f_func, 0.00001))
        print(format_time(t_vals), end="")
        flush()
except KeyboardInterrupt:
    print("Stopping test.")
else:
    print()

# Perform a step
print(format_title("Performing a step:"), end="")
try:
    for s in systems:
        t_vals = timer(step_func, args=(s, f_func, 0.00001))
        print(format_time(t_vals), end="")
        flush()
except KeyboardInterrupt:
    print("Stopping test.")
else:
    print()

# Performing a simulator step
print(format_title("Simulator step (no colls):"), end="")
sims = []
try:
    for s in systems:
        sim_ = sim.Simulation(s, f_func, 0.00001)
        t_vals = timer(sim_.step, kwargs={'collisions':False})
        print(format_time(t_vals), end="")
        sims.append(sim_)
        flush()
except KeyboardInterrupt:
    print("Stopping test.")
else:
    print()

# Checking collisions
print(format_title("Collision checking:"), end="")
try:
    for s in systems:
        t_vals = timer(s.get_collisions)
        print(format_time(t_vals), end="")
        flush()
except KeyboardInterrupt:
    print("Stopping test.")
except AttributeError:
    print("Unavailable.")
else:
    print()

# Testing collisions
print(format_title("Collision testing:"), end="")
try:
    for s in systems:
        t_vals = timer(s.kill_collisions, args=(test_func, null_f))
        print(format_time(t_vals), end="")
        flush()
except KeyboardInterrupt:
    print("Stopping test.")
except AttributeError:
    print("Unavailable.")
else:
    print()





# systems = [
#     sys_init_f(n) for n in SYS_SIZE
# ]

print("=======================")
