import pyglet

from args_parser import *
from controller import *

from sim import big_buffer

    # return s

window = pyglet.window.Window()
# def canvas_init():
#     window = pyglet.window.Window()
#     return window

p = [1, 1, 1]

@window.event
def on_draw():
    window.clear()

@window.event
def on_key_press(symbol, modifiers):
    print("A key was pressed")
    print("Symbol:", symbol)
    print("modifiers:", modifiers)

if __name__ == '__main__':
    # window = canvas_init()
    B, Sim = big_buffer()
    pyglet.app.run()
