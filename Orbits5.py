# from tkinter import *
import time
import loadSystem
import sys
# import os
sys.path.append('./simulation/')
import math

import numpy as np
import turtle
import matplotlib.pyplot as plt
import simulation.sim_funcs
from simulation.sim import Simulation, System
from camera import Camera
from controller import *
from turtle_graphics import *
from args_parser import *
import simulation.physics_functions
import loadSystem

from simulation.physics_functions import GravityNewtonian
FORCE_F = GravityNewtonian

from sim_funcs import leapfrog_init, leapfrog_step, RK4_init, RK4_step
from sim_funcs import kill_bounce
mi = get_arg_val("-mi")
if mi == 'leapfrog':
    INIT_F = leapfrog_init
    STEP_F = leapfrog_step
elif mi == 'RK4':
    INIT_F = RK4_init
    STEP_F = RK4_step

def main():
    print("Orbits5 Demonstration")
    from sim import big_buffer

    simulation.physics_functions.GRAVITATIONAL_CONSTANT = get_arg_val('-G')
    track_delta = False
    simulation.sim.DEFAULT_BUFFER_ATTRS += ['com']
    buffer_steps = 1
    cam_pos = 0
    cam_look = 0
    from simulation.sim import disc_sys

    if START_PAUSED:
        pause()
    if PRESET == '0':
        Sys = System([[5, .5, 0,], [-5, -.5, 0]], velocity=[[-1.0, 0, 0,], [1.0, 0, 0]], mass=1., radius=1.)
        Sim = Simulation(Sys, FORCE_F, t_step=get_arg_val('-d', 0.1))
        cam_pos = np.array([0, 0, 55])
        cam_look = np.array([0, 0, -1])
    elif PRESET == '1':
        # Simple 3 tier star-planet-moon solar system,
        # with a 4th stationary object falling into the middle.
        Sim = simple_system()
    elif PRESET == '2':
        kwargs = {
            'mass_props':(10, 0, 'normal'), 
            'radius_props':(get_arg_val("-dr", 10), 0, 'uniform'), 
            'particle_rad':(0.05, 0, 'normal'),
            'vel':('circ', FORCE_F), 
        }
        disc1 = disc_sys(PARTICLE_COUNT, **kwargs)
        disc2 = disc_sys(PARTICLE_COUNT, **kwargs, axis=[0, 0, 1])
        shift = get_arg_val("-dr", 10) * np.array([0, 1, 0])
        disc1.pos += shift
        disc2.pos -= shift
        disc1.vel += get_arg_val('-kick', 0) * np.array([10, 0, -10])

        disc1.append(disc2)
        disc1.initialize_info('force', 3, masked=True)
        # disc1.vel /= 1.5
        Sim = Simulation(disc1, FORCE_F, get_arg_val('-d', 0.005))
        Sim.track('force')
                
        # _, Sim = big_buffer(N=PARTICLE_COUNT, frames=500, radius_props=(15, 3, 'normal'),
        # particle_rad=(0.05, 0, 'normal'), mass_props=(10, 5, 'normal'), vel=(200, 1))
    elif PRESET == '3':
        # from simulation.sim import small_galaxy
        # Sim, _ = small_galaxy(N=PARTICLE_COUNT)
        kwargs = {
            'mass_props':(10, 0, 'normal'), 
            'radius_props':(get_arg_val("-dr", 10), 0, 'uniform'), 
            'particle_rad':(0.05, 0, 'normal'),
            'vel':('circ', FORCE_F), 
        }
        Sys = disc_sys(PARTICLE_COUNT, **kwargs)
        Sys.vel /= 1.2
        Sys.initialize_info('force', 3, 0., True)
        Sim = Simulation(Sys, FORCE_F, get_arg_val('-d', 0.01))
        Sim.track('force')
    elif PRESET == '4':
        # Rings around a planet
        Sim = rings(get_arg_val('-d', 0.005))
        cam_pos = [30, 0, 30]
        Sim.sys.initialize_info('force', 3, 0., True)
        Sim.track('force')
        # buffer_steps = 5
        # Sim.buffer(1, verb=True, n=buffer_steps, append_buffer=True)
    elif PRESET == '5':
        from simulation.sim import disc_merger
        Sys = disc_merger(
            PARTICLE_COUNT, thickness=0.5, disc_radii=5,
            vel=('circ', FORCE_F), mass_props=(20, 5, 'normal'),
            particle_rad=(0.03, 0.01, 'normal'), kick=np.asarray([1, 0, -1])*50,
            axis=[1, 0., 1], axis2=[1, 0., 1])
        Sys.vel /= 1.5
        Sys.initialize_info('force', 3, masked=True)
        Sim = Simulation(Sys, FORCE_F, t_step=get_arg_val('-d', 0.001))
        Sim.track('force')
    elif PRESET == '6':
        solar_sys_data = loadSystem.loadFile('systems/SolSystem.txt')
        pos = []
        vel = []
        mass = []
        rad = []
        get_pos = lambda x: [x['X'], x['Y'], x['Z']]
        get_vel = lambda x: [x['VX'], x['VY'], x['VZ']]
        get_mass = lambda x: x['MASS']
        get_density = lambda x: x['DENSITY']
        get_radius = lambda x: (3/(4 * np.pi)) * (get_mass(x) / get_density(x))**(1/3)
        id_map = {}; id = 0
        for d in solar_sys_data:
            x = solar_sys_data[d]
            if d[0] != '$':
                pos.append(get_pos(x))
                vel.append(get_vel(x))
                mass.append(get_mass(x))
                rad.append(get_radius(x))
                id_map[d] = id
                id += 1 
        earth_pos = get_pos(solar_sys_data['Earth'])
        moon_pos = get_pos(solar_sys_data['Moon'])
        cam_pos = .5 * (np.array(earth_pos) + np.array(moon_pos)) * 1e3
        pos = np.array(pos, dtype=float) * 1e3 # Convert from km to m
        vel = np.array(vel, dtype=float) * 1e3 
        mass = np.array(mass, dtype=float)
        rad = np.array(rad, dtype=float)
       	colours = {
            "Moon"		: [1,   1, 	 1  ],  # Photo realistic moon white
            "Earth"		: [0,   0.5, 1  ],  # Photo realistic ocean blue
            "Sun"		: [1,   1,   0  ],
            "Mercury"	: [1,   0.5, 0  ],
            "Venus"		: [1,   0.3, 0  ],
            "Mars"		: [1,   0,   0  ],
            "Jupiter"	: [1,   0.6, 0.2],
            "Saturn" 	: [1,   0.8, 0.5],
            "Uranus"	: [0.5, 0.5, 1  ],
            "Neptune"	: [0.2, 0.2, 1  ],
        }
        Sys = System(pos, vel, mass, rad)
        # Sys.initialize_info('force', 3, 0., True)
        Sys.initialize_info('colour', None, None, True)
        # Set colours:
        for i in id_map:
            if i in colours:
                Sys.colour[id_map[i]] = colours[i]
        Sim = Simulation(Sys, FORCE_F, get_arg_val('-d', 0.001))
        # Sim.track('force')
    elif PRESET=='7':
        min_dist = 6
        max_dist = 25
        star = [[0., 0., 0.]]; star_m = 1e4; star_rad = 4
        planet_count = get_arg_val('-n', 10)
        planet_dist = np.linspace(min_dist, max_dist, planet_count).reshape(-1, 1)
        planet_pos = np.array([[1., 0, 0,]]) * planet_dist
        pos = np.append(star, planet_pos, axis=0)
        mass = np.array([star_m, *([1e-2]*planet_count)])
        radius = np.array([star_rad, *([(max_dist - min_dist)/planet_count * 0.9/2]*planet_count)])
        Sys = System(pos, 0., mass, radius)
        for i in range(1, planet_count+1):
            simulation.physics_functions.circularise(Sys, i, 0, FORCE_F, [0, 0, 1])
        Sys.initialize_info('force', 3, 0., True)
        Sim = Simulation(Sys, FORCE_F, t_step=get_arg_val("-d", 0.001))
        Sim.track('force')
        cam_pos = np.array([2 * max(planet_dist), 0, 10])

    else:
        print(f"Preset {PRESET} does not exist.")
        return 0

    if get_arg_val('-bounce'):
        simulation.sim.FORCE_KILL_LOOP = True
        simulation.sim_funcs.bounce_damping = get_arg_val('-bdamp')   
        Sim.kill_func = simulation.sim_funcs.kill_bounce
    if get_arg_val('-spin'):
        Sim.sys.initialize_info('spin', 3, 0., True)
        # Sim.sys.vel *= 0.8
        if PRESET == '0':
            Sim.sys.set('spin', [0., 0., -1], 1)
            print(Sim.sys.spin)
            # Sim.spin[-1] = [0, -10.0, 0]
    Sim.sys.vel *= get_arg_val('-sm')


    # if arg_supplied('-d'):
    #     Sim.t_step = get_arg_val('-d')
    Sim.init_sim()
    if not np.any(cam_look):
        cam_look = np.array([-1., 0, -1])

    camera = Camera(Sim, pos=cam_pos + Sim.com, look=cam_look, screen_depth=1000)
    camera.set_X_Y_axes(new_Y = np.array([0., 1, 0]))

    # Call this to set the window up.
    canvas_init()

    turtle.listen()

    # initialise these for first step before they are set
    com_delta = 0.
    sleep = 0.
    top_shader = 10. # Used to keep a consistent shading spectrum
    while get_running() is not False:
        t_mult = get_time_mult()
        if t_mult == -1:
            Sim.sys.vel *= -1
            try:
                Sim.spin *= -1
            except:
                pass
        if sleep > 0: time.sleep(sleep)
        new_pause = get_pause()
        if new_pause != None:
            Sim.pause(new_pause)

        start = time.time()
        com_pre = Sim.com

        F = Sim.step(n=(buffer_steps if Sim.paused else 1), buffer_in_pause=get_arg_val('-bp'))

        com_delta = Sim.com - com_pre

        if track_delta:
            camera.pos += com_delta
            # print(camera.pos)
        render = camera.render(F)
        end = time.time()
        step_t = end-start

        new_pan = get_pan()
        new_rot = get_rotate()
        if np.any(camera.vel):
            new_pan = get_pan(True)
        if new_pan != None:
            camera.vel = 0.06 * new_pan[3] * camera.closest_surface_dist * (
                camera.screen_X_axis_norm * new_pan[0] +
                camera.screen_Y_axis_norm * new_pan[1] +
                camera.look * new_pan[2]
            )
        if new_rot != None:
            camera.rot = 0.04 *  (
                camera.screen_Y_axis_norm * new_rot[0] +
                camera.screen_X_axis_norm * new_rot[1] +
                camera.look * new_rot[2]
            )

        d_start = time.time() # drawing start time
        frame_clear()
        camera.step(1.)
        try:
            F_mag = (np.linalg.norm(Sim.force, 2, axis=-1))
            # F_mag /= Sim.mass # otherwise big bodies get too biased
            F_mag = np.sqrt(F_mag)
            F_max = max(F_mag)
            # top_shader = F_max
            # top_shader = np.mean(F_mag)
            # top_shader = F_max*.5 + top_shader*.5
            if F_max > top_shader:
                top_shader = F_max * 1.2
            if F_max < top_shader / 2:
                top_shader = 2 * F_max
        except:
            try:
                Sim.colour
            except:
                shading = [1, 0, 0]
            else:
                shading = Sim.colour
                shading[shading == None] = 'white'
        else:
            if F_max == 0:
                shading = [1, 0, 0]
            else:
                F_mag = F_mag / top_shader
                F_mag[F_mag > 1] = 1
                
                shading = np.array(
                    # [F_mag / top_shader, 1 - F_mag/top_shader, 0.*F_mag]
                    [F_mag, 1 - F_mag, 0.*F_mag]
                ).transpose()
        try:
            shading = Sim.spin * Sim.mass.reshape(-1, 1)
            # if np.any(shading != 0): shading /= np.max(np.abs(shading))
            shading = shading[:,2]
            shading = np.tanh(shading)
            shading = np.array([0.5+shading/2, 0.*shading, 0.5-shading/2]).transpose()
            # shading += 1
            # shading[:,0] = 1
        except Exception as e:
            pass
                # shading = shading.tolist()
        # print(shading) 
        if not (get_arg_val('-db') and Sim.paused):
            draw_all(*render, fill=shading)
        else:
            turtle.pencolor('white')
            turtle.write('BUFFERING (space to resume)')
        draw_t = time.time() - d_start
        if Sim.stored:
            ss = f'[{Sim.sys.N} active]'
        else:
            ss = ''
        data_str = f"""\r\
Buffer: {Sim.stored} Size: {Sim.N} {ss} \
Time: {1000*step_t:.3f}ms [+ {1000*draw_t:.3f}ms]"""
        print(f"{data_str: <105}", end = '')
        sys.stdout.flush()

        sleep = 0.02 - (step_t + draw_t)
        if (Sim.paused and get_arg_val('-db')): 
            # Don't sleep if nothing's on the screen
            sleep = 0
        frame_update()
    print()

    return Sim
        # print("f", end = '')

def simple_system():
    p1 = np.array([0, 0, 0], dtype=float)
    p2 = np.array([10, 0, 0], dtype=float)
    p3 = np.array([11, 0, 0], dtype=float)
    p4 = np.array([20, 0, 0], dtype=float)
    S = [0, 1, 2, 3]
    r  = np.array([1, 0.2, 0.01, 1])[S]
    m = np.array([100, 10, 1, 100])[S]
    Sys = System(np.array([p1, p2, p3, p4])[S], velocity=0.0, mass=m, radius=r)
    Sim = Simulation(Sys, FORCE_F, t_step=0.01, step_func=STEP_F, init_func=INIT_F)

    from physics_functions import circularise
    circularise(Sys, 1, 0, Sim.func, [0, 0, 1])
    circularise(Sys, 2, 1, Sim.func, [0, 0, 1])

    return Sim


def rings(t_step=0.0005):
    # from physics_functions import GravityNewtonian as FORCE_F
    simulation.physics_functions.GRAVITATIONAL_CONSTANT = get_arg_val('-G')    
    planet = [0., 0., 0.]; p_r = 10.
    moon   = [30., 0., 0.]; m_r = 0.5
    rand_angle = np.random.random(PARTICLE_COUNT) * np.pi * 2
    rand_dist  = np.random.random(PARTICLE_COUNT) * 30 + 15
    rand_p     = np.array([np.cos(rand_angle), 
                           np.sin(rand_angle), 
                           np.random.normal(scale=0.01, size=PARTICLE_COUNT)]).transpose()
    rand_p    *= rand_dist.reshape(-1, 1)
    mass = np.full(PARTICLE_COUNT+2, 1e-5)
    mass[0] = 1000
    mass[-1] = 1
    radius = np.full(PARTICLE_COUNT+2, 0.05)
    radius[0] = p_r
    radius[-1] = m_r

    C = simulation.physics_functions.circularise
    pos = np.array([planet, *rand_p, moon])
    Sys = System(pos, mass=mass, radius=radius, velocity=0.)
    Sim = Simulation(Sys, FORCE_F, t_step=t_step)
    C(Sys, -1, 0, Sim.func, [0, 0, 1])
    for i in range(1, PARTICLE_COUNT+1):
        C(Sys, i, 0, Sim.func, [0, 0, 1])
    # Sys.set_vel(0, np.array([0., 0., 0.]))
    Sys.pos -= Sys.vel[0].reshape(1, -1)
    # print(Sys.vel)

    return Sim

# def gas_cloud(N=500):


if __name__ == '__main__':
    s = main()
