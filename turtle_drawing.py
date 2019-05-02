import turtle
from args_parser import *

def canvas_init():
    window = turtle.Screen()
    window.setup(width = 1.0, height = 1.0)
    turtle.bgcolor([0, 0, 0])
    # turtle.bgcolor("white")

    turtle.tracer(0, 0)             # Makes the turtle's speed instantaneous
    turtle.hideturtle()


def frame_clear():
    turtle.clear()

def frame_update():
    turtle.update()


def draw_all(x, y, major, minor, angle, fill = [0, 0, 0], box = False, intensity = None, screen_radius=None):
    global ellipsePoints
    global drawStars
    global lowestApparentMag
    global totalIncidentIntensity
    global FLARE_POLY_POINTS
    if SMART_DRAW:
        perimApprox = 2*np.pi*np.sqrt((major**2 + minor**2) / 2)
        points = np.int32(perimApprox / SMART_DRAW_PARAMETER)
    else:
        points = np.full(len(x), ellipsePoints)

    points[points > MAX_POINTS] = MAX_POINTS
    # print('x', x)
    # print('y', y)
    # localX = major/2
    # localY = 0
    # screenX = localX * cos(angle) - localY * sin(angle)
    # screenY = localY * cos(angle) + localX * sin(angle)

    # coordAngle is the 'cartesian plane' angle of the centre of the oval
    # coordAngle = (0 if x == 0 else atan(y / x))
    # if (x < 0 and y < 0):
    #     coordAngle -= pi # Make negative
    # elif x < 0:
    #     coordAngle += pi # Make positive

    flareWidth = 5 # So it doesn't break later, normally this is 0
    centre_array = np.array([x, y]).transpose()

    # if (mag != None): mag += MAG_SHIFT
    if (intensity != None) and False:
        # This uses Rayleigh Criterion to determine the width of the diffraction 'flare'
        # ie, an 'Airy disk'. Using this it is then sufficient to set the intensity
        # of the centre to 100%, and the edge of the disk to 0%, and then have a linear
        # slope through the radius.

        # Rayleigh Criterion:
        # sin(x) = 1.22 lambda / d
        # x: angle from centre to first dark fringe
        # d: diameter of aperture
        # AIRY_COEFF is 1/log(AIRY_RATIO) so the log below becomes log base AIRY_RATIO

        # Generally the middle ring about x = 0 will be the only one visible, however when
        # the camera is close to very bright lights further maximums will be visible and overlap
        # (since we wouldn't be working with a point source for very bright sources).
        # Assume that each sucessive maximum is (AIRY_RATIO) the intensity of
        # the previous
        intensity = EXPOSURE * getIntensity(mag)
        tempDiffRadius = DIFFRACTION_RADIUS
        flareWidth = (AIRY_COEFF * log(MIN_VISIBLE_INTENSITY / intensity)) * tempDiffRadius

        if (lowestApparentMag == None):
            lowestApparentMag = mag
        elif (lowestApparentMag and mag < lowestApparentMag):
            lowestApparentMag = mag
        elif (points <= 2 and mag == None):
            fill = [1, 1, 1]

    if box and False:
        # box 'inner' radius, ie half the width of a side
        boxRadius = max(MIN_BOX_WIDTH, major * 1.4 + flareWidth) / 2
        turtle.up()
        turtle.pencolor([1, 1, 1])
        turtle.goto(x - boxRadius, y - boxRadius)
        turtle.down()
        turtle.goto(x - boxRadius, y + boxRadius)
        turtle.goto(x + boxRadius, y + boxRadius)
        turtle.goto(x + boxRadius, y - boxRadius)
        turtle.goto(x - boxRadius, y - boxRadius)
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

        turtle.fillcolor('red')
        turtle.pencolor('red')
        # onScreen = True
        # Drawn = False
        # draw between <angle> and <np.pi - angle>
        clipAngle = np.pi/2 - clipAngle
        start = centre_array + localShift(clipAngle)


        ### NUMPY TAKING OVER FROM HERE
        # print(start)
        for j in range(len(x)):
            s = start[j]
            c = centre_array[j]
            c_a = clipAngle[j]

            # print(s)

            turtle.goto(*s)
            turtle.begin_fill()

            start_i = 0
            end_i = points[j]
            if end_i <= 2:
                turtle.dot(2)
            else:
                for i in range(start_i, end_i):
                    # if (i == start): turtle.dot(10)
                    tempAngle = 2*((end_i / 2) - i)/end_i * c_a
                    point = c + localShift(tempAngle, j)
                    turtle.goto(*point)

            turtle.end_fill() # Draw the oval:
            turtle.up()
    else:
        turtle.up()
        turtle.goto()

    # if (drawStars):
    #     turtle.up()
    #     turtle.goto(x, y)
    #     turtle.pencolor(fill)
    #     if (intensity == None):
    #         if (points < 2):
    #             turtle.dot(2)
    #         return True
    #
    #     # Scale up fill:
    #     if type(fill) == list:
    #         M = max(fill)
    #         fill = [x * 1/M for x in fill]
    #
    #     flareBuffer = [centre]
    #
    #     if flareWidth > MAX_RINGS:
    #         step = -flareWidth / MAX_RINGS
    #     else:
    #         step = -1
    #     # for r in range(int(flareWidth), 0, step):
    #     r = flareWidth
    #     while (r > 0):
    #         # Ir: Intensity at radius 'r' (Scaled so that 0 is the minimum threshold)
    #         # Ir = intensity * (1 - r / flareWidth)
    #         scale = (1 - r / flareWidth) ** 2
    #         if (scale < MIN_RMAG):
    #             r += step
    #         # continue
    #     if not DIFF_SPIKES:
    #         # turtle.pencolor([x * scale for x in fill])
    #         # turtle.dot((r) + minor)
    #         flareBuffer.append([r + major, [x * scale for x in fill]])
    #     else:
    #         polydot(r + major, [x * scale for x in fill])
    #     # drawOval(x, y, major + r, minor + r, angle, fill = [x * scale for x in fill])
    #         r += step
    #     global fullFlareBuffer
    #     fullFlareBuffer.append(flareBuffer)
    #     # else:
    #     #     return False
    #     totalIncidentIntensity += intensity
    return True


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
#
# turtle.onkey(cycleTargets, "Tab")
# turtle.onkeypress(togglePanTrack, "t")
# turtle.onkey(toggleRotTrack, "y")
# turtle.onkey(clearTarget,    "c")
# turtle.onkey(goToTarget,     "g")
# turtle.onkey(toggleRealTime, "i")
#
# turtle.onkey(toggleScreenData, "h")
# turtle.onkey(startRecord, "j")
# turtle.onkey(stopRecord, "k")
#
# turtle.onkeypress(upScreenDepth, "'")
# turtle.onkeypress(downScreenDepth, ";")
#
# turtle.onkeypress(upMaxMag, ".")
# turtle.onkeypress(downMaxMag, ",")
#
# turtle.onkey(upDelta, "]")
# turtle.onkey(downDelta, "[")
# turtle.onkey(revDelta, "\\")
#
# turtle.onscreenclick(leftClick, 1)
# turtle.onscreenclick(rightClick, 3)
#
# turtle.onkey(bufferRecord, "n")
# turtle.onkey(bufferPlay, "m")
#
# turtle.onkey(upRelSpeed, "=")
# turtle.onkey(dnRelSpeed, "-")
#
# turtle.onkey(search, "/")
