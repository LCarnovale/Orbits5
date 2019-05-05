# from tkinter import *
import time
import loadSystem
import sys

import numpy as np
import turtle
import matplotlib.pyplot as plt
from sim import Simulation, System
from camera import Camera
from controller import *

from turtle_graphics import *
from args_parser import *
import physics_functions

from physics_functions import GravityNewtonian
FORCE_F = GravityNewtonian
from sim_funcs import leapfrog_init, leapfrog_step
INIT_F = leapfrog_init
STEP_F = leapfrog_step

def main():
    print("Orbits5 Demonstration")
    canvas_init()

    from sim import big_buffer
    physics_functions.GRAVITATIONAL_CONSTANT = args['-G'][1]
    track_delta = False
    if START_PAUSED:
        pause()
    if PRESET == '1':
        Sim = simple_system()
        Sim.buffer(1)
        track_delta = True
    elif PRESET == '2':
        _, Sim = big_buffer(N=PARTICLE_COUNT, frames=500)
        track_delta = True
    elif PRESET == '3':
        from sim import small_galaxy
        Sim, Sys = small_galaxy(N=PARTICLE_COUNT)
        track_delta = True
    elif PRESET == '4':
        Sim = rings()
        Sim.buffer(300, verb=True, n=4, append_buffer=True)
        track_delta = True
        # Rings around a planet
    if args['-d'][-1]:
        Sim.t_step = args['-d'][1]

    camera = Camera(Sim, pos=np.array([40., 0, 40]), look=np.array([-1., 0, -1]), screen_depth=1000)
    camera.set_X_Y_axes(new_Y = np.array([-1., 0, 1]))


    turtle.listen()
    # print(F._dict)
    # F = False
    CoM = Sim.sys.mass.reshape(-1, 1) * Sim.sys.pos
    CoM = np.mean(CoM, axis=0)

    while get_running() is not False:
        time.sleep(0.02)
        # print(Sim.sys.N)
        new_pause = get_pause()
        if new_pause != None:
            print(f"Pause = {Sim.paused}")
            Sim.pause(new_pause)

        frame_clear()
        start = time.time()
        # if F:
        # render = camera.render(F)
            # if not Sim.paused:
        # F = B.pull()
        # else:
        F = Sim.step()
        if F:
            # Sim.step_collisions()
            render = camera.render(F)
        else:
            render = camera.render()
        end = time.time()
        print(f"\
Buffer: {Sim.stored} Size: {Sim.sys.N} \
Time: {1000*(end-start):.3f} ms"+' '*20, end = '\r')
        sys.stdout.flush()
        # camera.look_at(lock)

        draw_all(*render)
        new_pan = get_pan()
        new_rot = get_rotate()
        if new_rot != None:
            camera.rot = 0.02  *  (
                camera.screen_Y_axis_norm * new_rot[0] +
                camera.screen_X_axis_norm * new_rot[1] +
                camera.look * new_rot[2]
            )
        if np.any(camera.vel):
            new_pan = get_pan(True)
        if new_pan != None:
            # render = camera.render()
            camera.vel = 0.06 * new_pan[3] * camera.closest_surface_dist * (
                camera.screen_X_axis_norm * new_pan[0] +
                camera.screen_Y_axis_norm * new_pan[1] +
                camera.look * new_pan[2]
            )
        # Track centre of mass:
        # camera.pos -= CoM.copy()
        # CoM = Sim.sys.mass.reshape(-1, 1) * Sim.sys.pos
        # CoM = np.mean(CoM, axis=0)
        if track_delta:
            camera.pos += np.mean(
                Sim.pos_delta * Sim.mass.reshape(-1, 1) / np.sum(Sim.mass), 
            axis=0)

        camera.step(1.)
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
    m = np.array([100, 10, 1, 0.2])[S]
    Sys = System(np.array([p1, p2, p3, p4])[S], velocity=0.0, mass=m, radius=r)
    Sim = Simulation(Sys, FORCE_F, t_step=0.01, step_func=STEP_F, init_func=INIT_F)

    from physics_functions import circularise
    circularise(Sys, 1, 0, Sim.func, [0, 0, 1])
    circularise(Sys, 2, 1, Sim.func, [0, 0, 1])

    return Sim


def rings():
    physics_functions.GRAVITATIONAL_CONSTANT = args['-G'][1]
    # from physics_functions import GravityNewtonian as FORCE_F
    planet = [0., 0., 0.]; p_r = 10.
    rand_angle = np.random.random(PARTICLE_COUNT) * np.pi * 2
    rand_dist  = np.random.random(PARTICLE_COUNT) * 10 + 13
    rand_p     = np.array([np.cos(rand_angle), np.sin(rand_angle), np.random.normal(scale=0.01, size=PARTICLE_COUNT)]).transpose()
    rand_p    *= rand_dist.reshape(-1, 1)
    mass = np.full(PARTICLE_COUNT+1, 100. / PARTICLE_COUNT)
    mass[0] = 400
    radius = np.full(PARTICLE_COUNT+1, 0.05)
    radius[0] = p_r

    C = physics_functions.circularise
    pos = np.array([planet, *rand_p])
    Sys = System(pos, mass=mass, radius=radius, velocity=0.)
    Sim = Simulation(Sys, FORCE_F, t_step=0.001)
    for i in range(1, PARTICLE_COUNT+1):
        C(Sys, i, 0, Sim.func, [0, 0, 1])
    Sys.pos[0] *= 0.
    # print(Sys.vel)

    return Sim

# def gas_cloud(N=500):


if __name__ == '__main__':
    s = main()
