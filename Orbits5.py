from tkinter import *
import time
import loadSystem
import sys

import numpy as np
import turtle
import matplotlib.pyplot as plt
from sim import Simulation, System
from camera import Camera
from controller import *

from turtle_drawing import *
from args_parser import *
import physics_functions

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
        B = Sim.buffer(1)
    elif PRESET == '2':
        B, Sim = big_buffer(N=PARTICLE_COUNT, frames=500)
        track_delta = True
    elif PRESET == '3':
        from sim import small_galaxy
        Sim, Sys = small_galaxy(N=PARTICLE_COUNT)
        if args['-d'][-1]:
            Sim.t_step = args['-d'][1]
        # B = Sim.buffer(50, n=4, verb=True)
    elif PRESET == '4':
        Sim = rings()
        B = Sim.buffer(300, verb=True, n=4, append_buffer=True)
        track_delta = True
        # Rings around a planet

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
        print(f"Buffer: {Sim.stored} Size: {Sim.sys.N} Time: {1000*(end-start):.3f} ms   ", end = '\r'); sys.stdout.flush()
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
        if track_delta: camera.pos += np.mean(Sim.sys.pos_delta, axis=0)

        # print(camera.vel)
        # delta = Sim.sys.pos_delta[lock]
        # camera.pos += delta
        # camera.vel += Sim.sys.vel[lock]
        camera.step(1.)
        frame_update()
    print()
    return Sim
        # print("f", end = '')

def simple_system():
    from physics_functions import GravityNewtonian as gfunc
    p1 = np.array([0, 0, 0], dtype=float)
    p2 = np.array([10, 0, 0], dtype=float)
    p3 = np.array([11, 0, 0], dtype=float)
    p4 = np.array([20, 0, 0], dtype=float)
    S = [0, 1, 2, 3]
    r  = np.array([1, 0.2, 0.01, 1])[S]
    m = np.array([100, 10, 1, 0.2])[S]
    Sys = System(np.array([p1, p2, p3, p4])[S], velocity=0.0, mass=m, radius=r)
    Sim = Simulation(Sys, gfunc, t_step=0.01)

    from physics_functions import circularise
    circularise(Sys, 1, 0, Sim.func, [0, 0, 1])
    circularise(Sys, 2, 1, Sim.func, [0, 0, 1])

    return Sim


def rings():
    physics_functions.GRAVITATIONAL_CONSTANT = args['-G'][1]
    from physics_functions import GravityNewtonian as gfunc
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
    Sim = Simulation(Sys, gfunc, t_step=0.001)
    for i in range(1, PARTICLE_COUNT+1):
        C(Sys, i, 0, Sim.func, [0, 0, 1])
    Sys.pos[0] *= 0.
    # print(Sys.vel)

    return Sim





# Running = True
# rotate = [0, 0, 0]
# pan = [0, 0, 0, 1]
# shiftL = False
#
# def panRight():
# 	if pan[0] < 1:
# 		pan[0] += 1
#
# def panLeft():
# 	if pan[0] > - 1:
# 		pan[0] -= 1
#
# def panBack():
# 	if pan[2] > - 1:
# 		pan[2] -= 1
#
# def panForward():
# 	if pan[2] < 1:
# 		pan[2] += 1
#
# def panDown():
# 	if pan[1] > - 1:
# 		pan[1] -= 1
#
# def panUp():
# 	if pan[1] < 1:
# 		pan[1] += 1
#
# def panFast():
# 	global shiftL
# 	shiftL = True
# 	pan[3] = 15
#
# def panSlow():
# 	global shiftL
# 	shiftL = False
# 	pan[3] = 1
#
# def rotRight():
# 	if rotate[0] < 1:
# 		rotate[0] = rotate[0] + 1
#
# def rotLeft():
# 	if rotate[0] > -1:
# 		rotate[0] = rotate[0] - 1
#
# def rotDown():
# 	if rotate[1] < 1:
# 		rotate[1] += 1
#
# def rotUp():
# 	if rotate[1] > -1:
# 		rotate[1] -= 1
#
# def rotAntiClock():
# 	if rotate[2] < 1:
# 		rotate[2] += 1
#
# def rotClockWise():
# 	if rotate[2] > -1:
# 		rotate[2] -= 1
#
# def escape():
# 	global Running
# 	Running = False
#
# turtle.onkeypress(panLeft, "a")
# turtle.onkeyrelease(panRight , "a")
#
# turtle.onkeypress(panRight, "d")
# turtle.onkeyrelease(panLeft , "d")
#
# turtle.onkeypress(panForward, "w")
# turtle.onkeyrelease(panBack , "w")
#
# turtle.onkeypress(panBack, "s")
# turtle.onkeyrelease(panForward , "s")
#
# turtle.onkeypress(panUp, "r")
# turtle.onkeyrelease(panDown , "r")
#
# turtle.onkeypress(panDown, "f")
# turtle.onkeyrelease(panUp , "f")
#
# turtle.onkeypress(panFast, "Shift_L")
# turtle.onkeyrelease(panSlow, "Shift_L")
#
# turtle.onkeypress(rotRight, "Right")
# turtle.onkeyrelease(rotLeft, "Right")
#
# turtle.onkeypress(rotLeft, "Left")
# turtle.onkeyrelease(rotRight, "Left")
#
# turtle.onkeypress(rotUp, "Up")
# turtle.onkeyrelease(rotDown, "Up")
#
# turtle.onkeypress(rotDown, "Down")
# turtle.onkeyrelease(rotUp, "Down")
#
# turtle.onkeypress(rotClockWise, "e")
# turtle.onkeyrelease(rotAntiClock, "e")
#
# turtle.onkeypress(rotAntiClock, "q")
# turtle.onkeyrelease(rotClockWise, "q")
#
# turtle.onkey(escape, "Escape")
# turtle.onkey(pause,  "space")


if __name__ == '__main__':
    s = main()
