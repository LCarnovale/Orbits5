# Functions to handle controlling the simulation

controller_flags = {
    'sim_paused': False,
    'sim_running': True,
    'rotate': [0, 0, 0],
    'pan': [0, 0, 0, 1],
    'shiftL': False
}

change_flags = {f:False for f in controller_flags}

def get_flag_if_changed(flag, ignore=False):
    """
    General function to return the value of a flag only
    if it has changed since last being 'gotten'.

    If ignore is True, then the value is returned regardless of it
    has been changed.
    """
    # out = (change_flags[flag] or None) and controller_flags[flag]
    # if out is not None: change_flags[flag] = False
    out = controller_flags[flag]
    if change_flags[flag] or ignore:
        change_flags[flag] = False
        return out
    else:
        return None
    # if change_flags[flag] or ignore:
    #     change_flags[flag] = False
    #     return controller_flags[flag]
    # else:
    #     return None

def set_flag(flag, value):
    """
    General function to set the value of a flag,
    will also set the relevant change_flag to True.
    """
    change_flags[flag] = True
    controller_flags[flag] = value

# These functions should be used by the simulator to quickly be
# able to tell if a value has changed and if so get the new value.
# The optional argument ignore can be set to True to have
# the function always return the value.
def get_pan(ignore=False):
    return get_flag_if_changed('pan', ignore)

def get_rotate(ignore=False):
    return get_flag_if_changed('rotate', ignore)

def get_pause(ignore=False):
    return get_flag_if_changed('sim_paused', ignore)

def get_running(ignore=False):
    return get_flag_if_changed('sim_running', ignore)




def panRight():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    if pan[0] < 1:
        pan[0] += 1
    # set_flag('pan', pan)


def panLeft():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    if pan[0] > - 1:
        pan[0] -= 1

def panBack():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    if pan[2] > - 1:
        pan[2] -= 1

def panForward():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    if pan[2] < 1:
        pan[2] += 1

def panDown():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    if pan[1] > - 1:
        pan[1] -= 1

def panUp():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    if pan[1] < 1:
        pan[1] += 1

def panFast():
    pan = controller_flags['pan']
    change_flags['pan'] = True
    set_flag('shiftL', True)
    pan[3] = 15

def panSlow():
    global shiftL
    pan = controller_flags['pan']
    change_flags['pan'] = True
    set_flag('shiftL', False)
    pan[3] = 1

def rotRight():
    rotate = controller_flags['rotate']
    if rotate[0] > -1:
        rotate[0] = rotate[0] - 1
    change_flags['rotate'] = True

def rotLeft():
    rotate = controller_flags['rotate']
    if rotate[0] < 1:
        rotate[0] = rotate[0] + 1
    change_flags['rotate'] = True

def rotDown():
    rotate = controller_flags['rotate']
    if rotate[1] > -1:
        rotate[1] -= 1
    change_flags['rotate'] = True

def rotUp():
    rotate = controller_flags['rotate']
    if rotate[1] < 1:
        rotate[1] += 1
    change_flags['rotate'] = True

def rotAntiClock():
    rotate = controller_flags['rotate']
    if rotate[2] < 1:
        rotate[2] += 1
    change_flags['rotate'] = True

def rotClockWise():
    rotate = controller_flags['rotate']
    if rotate[2] > -1:
        rotate[2] -= 1
    change_flags['rotate'] = True

def escape():
    set_flag('sim_running', False)

def pause():
    set_flag('sim_paused', not controller_flags['sim_paused'])

#
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
