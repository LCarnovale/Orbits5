from tkinter import *
import time
import loadSystem

import numpy as np
# import turtle
import turtle_drawing
import matplotlib.pyplot as plt
from sim import Simulation, System
from camera import Camera

from args_parser import *

def main():
    print("Orbits5 Demonstration")

    from sim import big_buffer
    B = big_buffer(500, 1000)
    Sim = simple_system()
    turtle_drawing.canvas_init()
    camera = Camera(Sim.sys, pos=np.array([15, 0, 15])*15, look=np.array([-1, 0, -1]), screen_depth=10000)
    camera.set_X_Y_axes(new_Y = np.array([-1, 0, 1]))

    # B = Sim.buffer(500)
    # for i in range(5000):
    for i in range(1000):
        turtle_drawing.frame_clear()
        render = camera.render(B, i)
        # Sim.step()
        # render = camera.render()
        turtle_drawing.draw_all(*render)
        turtle_drawing.frame_update()

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




if __name__ == '__main__':
    main()
