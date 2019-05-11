# The graphics module for drawing a rendering of a simulation.
# This module should import functions from the controller
# and from args_parser as necessary, and link relevant functions
# to user controls


import turtle
from math import sqrt

import numpy as np

from args_parser import *
from controller import *


_window = None

def screen_width():
    try:
        return _window.window_width()
    except:
        return None # Raise exception?
    
def screen_height():
    try:
        return _window.window_height()
    except:
        return None # Raise exception?
    

def canvas_init():
    global _window
    window = turtle.Screen()
    turtle.hideturtle()
    window.setup(width = 1.0, height = 1.0)
    window.bgcolor([0, 0, 0])
    window.tracer(0, 0)             # Makes the turtle's speed instantaneous
    bind_input()
    _window = window
    return window


def frame_clear():
    turtle.clear()

def frame_update():
    turtle.update()

def play_buffer(buffer):
    """
    Open a window and play a buffer
    """
    canvas_init()


def _is_array(x):
    # Returns true if x is an N x M or deeper array,
    # ie it has atleast 2 dimensions,
    # else False
    try:
        a = np.shape(x)
        if len(a) >= 2:
            return True    
        # else:
        #     print(a)
    except:
        return False
    else:
        return False
    # finally:
    #     return False # funny


def draw_all(x, y, z, major, minor, angle, fill = [0,0,0],
            box = None, intensity = None, screen_radius=None):
    global ellipsePoints
    global drawStars
    global lowestApparentMag
    global totalIncidentIntensity
    global FLARE_POLY_POINTS

    mask = (z > 0)
    mask = np.logical_and(mask, np.abs(x) < screen_width()/2)
    mask = np.logical_and(mask, np.abs(y) < screen_height()/2)
    sort_mask = np.argsort(z)[::-1]
    mask = mask[sort_mask]
    
    if box:
        if mask[box]:
            box_x = x[box]
            box_y = y[box]
            box_maj = major[box]
            # box_min = minor[box]
        else:
            box = None

    x = x[sort_mask][mask]
    y = y[sort_mask][mask]
    major = major[sort_mask][mask]
    minor = minor[sort_mask][mask]
    angle = angle[sort_mask][mask]
    # mask_map = np.flatnonzero(mask) # Indexes of True's in mask
    if _is_array(fill):
        fill = fill[sort_mask][mask]
    
    if SMART_DRAW:
        perimApprox = 2*np.pi*np.sqrt((major**2 + minor**2) / 2)
        points = np.int32(perimApprox / SMART_DRAW_PARAMETER)
    else:
        points = np.full(len(x), ellipsePoints)

    points[points > MAX_POINTS] = MAX_POINTS

    flareWidth = 5 # So it doesn't break later, normally this is 0
    centre_array = np.array([x, y]).transpose()

    if box:
        # box 'inner' radius, ie half the width of a side
        boxRadius = max(MIN_BOX_WIDTH, box_maj * 1.4 + flareWidth) / 2
        turtle.up()
        turtle.pencolor([1, 1, 1])
        turtle.goto(box_x - boxRadius, box_y - boxRadius)
        turtle.down()
        turtle.goto(box_x - boxRadius, box_y + boxRadius)
        turtle.goto(box_x + boxRadius, box_y + boxRadius)
        turtle.goto(box_x + boxRadius, box_y - boxRadius)
        turtle.goto(box_x - boxRadius, box_y - boxRadius)
        turtle.up() #Draw the box



    if flareWidth < 0: return False

    if False and (points > 2) and screen_radius:
        # screenRadius
        # Assuming that the angle puts the major axs through
        # the screen centre we can clip points that are outside
        # of the screen:
        centreRadius = (x**2 + y**2)**(1/2)
        X = centreRadius - screenRadius
        if (X <= -major or points < 10):
            # The whole ellipse is on screen.
            # or the thing is small anyway
            clipAngle = -np.pi/2
        elif (X >= major):
            # The whole ellipse is probably off the screen
            return
        else:
            # Do some quik mafs:
            Y = minor/major * sqrt(major**2 - X**2)
            cosTheta = Y / sqrt(X**2 + Y**2)
            clipAngle = -acos(cosTheta)
            # 'clipAngle' gives the angle from the minor axis
            # that the perimeter clips the screens radius
            # +np.pi/2 ==> the centre end of the perimeter is
            #        just touching the screen radius
            #        (so the whole thing is probably out of view)
            # -np.pi/2 ==> the outer end is just touching,
                    #        ie, almost all of it will be in view.
    turtle.up()
    if np.any(points > 2) or True:
        clipAngle = np.full(len(x), -np.pi / 2)

        # Shifts an xy pair relative to major-minor axes
        # to the x-y axes of the screen, returns the new xy pair relative
        # to the centre of the oval and the screen x-y plane
        def localShift(local_angle, index=slice(None)):
            # coordAngle -= pi
            locX = major[index]/2 * np.cos(local_angle)
            locY = minor[index]/2 * np.sin(local_angle)
            shiftX = locX * np.cos(angle[index]-np.pi) - locY * np.sin(angle[index]-np.pi)
            shiftY = locY * np.cos(angle[index]-np.pi) + locX * np.sin(angle[index]-np.pi)
            return np.array([shiftX, shiftY]).transpose()

        # onScreen = True
        # Drawn = False
        # draw between <angle> and <np.pi - angle>
        clipAngle = np.pi/2 - clipAngle
        start = centre_array + localShift(clipAngle)


        ### NUMPY TAKING OVER FROM HERE
        # print(start)
        try:
            fill = fill.tolist()
        except:
            pass
        for j in range(len(x)):
            try:
                fill[0][0]
            except:
                fill_ = fill
            else:
                try:
                    fill_ = fill[j]
                except:
                    # print(fill)
                    raise KeyError

            turtle.fillcolor(fill_)
            turtle.pencolor(fill_)
            s = start[j]
            c = centre_array[j]
            c_a = clipAngle[j]

            # print(s)

            turtle.goto(*s)

            start_i = 0
            end_i = points[j]
            if end_i <= 2:
                turtle.dot(2)
            else:
                turtle.begin_fill()
                for i in range(start_i, end_i):
                    tempAngle = 2*((end_i / 2) - i)/end_i * c_a
                    point = c + localShift(tempAngle, j)
                    turtle.goto(*point)
                turtle.end_fill() # Draw the oval
            # if not j % 10:     
            #     turtle.write(f'fill: {fill_}')

            turtle.up()
    # else:
    #     turtle.up()
    #     turtle.goto()

    return True


def bind_input():
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
    #
    turtle.onkey(cycleTargets, "Tab")
    turtle.onkeypress(togglePanTrack, "t")
    turtle.onkey(toggleRotTrack, "y")
    turtle.onkey(clearTarget,    "c")
    turtle.onkey(goToTarget,     "g")
    # turtle.onkey(toggleRealTime, "i")
    #
    # turtle.onkey(toggleScreenData, "h")
    # turtle.onkey(startRecord, "j")
    # turtle.onkey(stopRecord, "k")
    #
    turtle.onkeypress(upScreenDepth, "'")
    turtle.onkeypress(downScreenDepth, ";")
    #
    # turtle.onkeypress(upMaxMag, ".")
    # turtle.onkeypress(downMaxMag, ",")
    #
    # turtle.onkey(upDelta, "]")
    # turtle.onkey(downDelta, "[")
    turtle.onkey(reverse_time, "\\")
    #
    turtle.onscreenclick(leftClick, 1)
    turtle.onscreenclick(rightClick, 3)
    #
    # turtle.onkey(bufferRecord, "n")
    # turtle.onkey(bufferPlay, "m")
    #
    # turtle.onkey(upRelSpeed, "=")
    # turtle.onkey(dnRelSpeed, "-")
    #
    # turtle.onkey(search, "/")
