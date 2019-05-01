from tkinter import *
import time
import loadSystem

import numpy as np
import turtle
import turtle_drawing
import matplotlib.pyplot as plt
from sim import Simulation, System
from camera import Camera
from controller import *

from args_parser import *

def main():
    print("Orbits5 Demonstration")
    turtle_drawing.canvas_init()

    from sim import big_buffer
    Sim = simple_system()
    B = Sim.buffer(1)
    B, Sim = big_buffer(N=PARTICLE_COUNT, frames=1)

    camera = Camera(Sim.sys, pos=np.array([15., 0, 15]), look=np.array([-1., 0, -1]), screen_depth=1000)
    camera.set_X_Y_axes(new_Y = np.array([-1., 0, 1]))


    turtle.listen()
    F = False
    # F = B.pull()
    while get_running() != False:
        # print(Sim.sys.N)
        new_pause = get_pause()
        if new_pause != None:
            Sim.pause(new_pause)

        turtle_drawing.frame_clear()
        camera.step(1.)
        camera.look_at(0)
        if F:
            render = camera.render(F)
            F = B.pull()
            # print(F)
        else:
            Sim.step()
            Sim.step_collisions()
            render = camera.render()
        turtle_drawing.draw_all(*render)
        new_pan = get_pan()
        new_rot = get_rotate()
        if new_rot != None:
            camera.rot = 0.02  *  (
                camera.screen_Y_axis_norm * new_rot[0] +
                camera.screen_X_axis_norm * new_rot[1] +
                camera.look * new_rot[2]
            )
        if new_rot and np.any(camera.vel):
            new_pan = get_pan(True)
        if new_pan != None:
            # render = camera.render()
            camera.vel = 0.06 * new_pan[3] *(
                camera.screen_X_axis_norm * new_pan[0] +
                camera.screen_Y_axis_norm * new_pan[1] +
                camera.look * new_pan[2]
            )
        # print(camera.vel)
        turtle_drawing.frame_update()
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

turtle.onkeypress(panLeft, "a")
turtle.onkeyrelease(panRight , "a")

turtle.onkeypress(panRight, "d")
turtle.onkeyrelease(panLeft , "d")

turtle.onkeypress(panForward, "w")
turtle.onkeyrelease(panBack , "w")

turtle.onkeypress(panBack, "s")
turtle.onkeyrelease(panForward , "s")

turtle.onkeypress(panUp, "r")
turtle.onkeyrelease(panDown , "r")

turtle.onkeypress(panDown, "f")
turtle.onkeyrelease(panUp , "f")

turtle.onkeypress(panFast, "Shift_L")
turtle.onkeyrelease(panSlow, "Shift_L")

turtle.onkeypress(rotRight, "Right")
turtle.onkeyrelease(rotLeft, "Right")

turtle.onkeypress(rotLeft, "Left")
turtle.onkeyrelease(rotRight, "Left")

turtle.onkeypress(rotUp, "Up")
turtle.onkeyrelease(rotDown, "Up")

turtle.onkeypress(rotDown, "Down")
turtle.onkeyrelease(rotUp, "Down")

turtle.onkeypress(rotClockWise, "e")
turtle.onkeyrelease(rotAntiClock, "e")

turtle.onkeypress(rotAntiClock, "q")
turtle.onkeyrelease(rotClockWise, "q")

turtle.onkey(escape, "Escape")
turtle.onkey(pause,  "space")


if __name__ == '__main__':
    main()
