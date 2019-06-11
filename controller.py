# Functions to handle controlling the simulation

# Define some flags that can get imported into main modules:
_controller_flags = {
    'sim_paused': False,
    'sim_running': True,
    'rotate': [0, 0, 0],
    'pan': [0, 0, 0, 1],
    'buffering': False,
    'drawing': True,
    'shiftL': False
}

_change_flags = {f:False for f in _controller_flags}

_cycle_target = None
_tracking_pos = None
_tracking_look = None
_click_store = []
# When changing, multiply current screen depth to get
# the new one desired by user
_screen_depth_multiplier = 1 
_time_step_multiplier = 1
def get_time_mult():
    global _time_step_multiplier
    out = _time_step_multiplier
    _time_step_multiplier = 1
    return out


def get_flag_if_changed(flag, ignore=False):
    """
    General function to return the value of a flag only
    if it has changed since last being 'gotten'.

    If ignore is True, then the value is returned regardless of it
    has been changed.
    """
    out = _controller_flags[flag]
    if _change_flags[flag] or ignore:
        _change_flags[flag] = False
        return out
    else:
        return None

def set_flag(flag, value):
    """
    General function to set the value of a flag,
    will also set the relevant change_flag to True.
    """
    _change_flags[flag] = True
    _controller_flags[flag] = value

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

# def get_


def panRight():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    if pan[0] < 1:
        pan[0] += 1
    # set_flag('pan', pan)


def panLeft():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    if pan[0] > - 1:
        pan[0] -= 1

def panBack():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    if pan[2] > - 1:
        pan[2] -= 1

def panForward():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    if pan[2] < 1:
        pan[2] += 1

def panDown():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    if pan[1] > - 1:
        pan[1] -= 1

def panUp():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    if pan[1] < 1:
        pan[1] += 1

def panFast():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    set_flag('shiftL', True)
    pan[3] = 15

def panSlow():
    pan = _controller_flags['pan']
    _change_flags['pan'] = True
    set_flag('shiftL', False)
    pan[3] = 1

def rotRight():
    rotate = _controller_flags['rotate']
    if rotate[0] > -1:
        rotate[0] = rotate[0] - 1
    _change_flags['rotate'] = True

def rotLeft():
    rotate = _controller_flags['rotate']
    if rotate[0] < 1:
        rotate[0] = rotate[0] + 1
    _change_flags['rotate'] = True

def rotDown():
    rotate = _controller_flags['rotate']
    if rotate[1] > -1:
        rotate[1] -= 1
    _change_flags['rotate'] = True

def rotUp():
    rotate = _controller_flags['rotate']
    if rotate[1] < 1:
        rotate[1] += 1
    _change_flags['rotate'] = True

def rotAntiClock():
    rotate = _controller_flags['rotate']
    if rotate[2] < 1:
        rotate[2] += 1
    _change_flags['rotate'] = True

def rotClockWise():
    rotate = _controller_flags['rotate']
    if rotate[2] > -1:
        rotate[2] -= 1
    _change_flags['rotate'] = True

def escape():
    set_flag('sim_running', False)

def pause():
    set_flag('sim_paused', not _controller_flags['sim_paused'])

    
def cycleTargets():
    global _cycle_target
    _cycle_target = True
    pass

def toggleRotTrack():
    pass

def clearTarget():
    pass

def goToTarget():
    pass

def togglePanTrack():
    pass

def get_click():
    # Retrieve and remove the earliest unseen click.
    global _click_store
    out = _click_store.pop(0)
    return out

# 0 for left click, 1 for right
def leftClick(x, y):
	_click_store.append([x, y, 0])

def rightClick(x, y):
	_click_store.append([x, y, 1])

def get_screen_depth_mult():
    global _screen_depth_multiplier
    out = _screen_depth_multiplier
    _screen_depth_multiplier = 1
    return out

def upScreenDepth():
    global _screen_depth_multiplier
    _screen_depth_multiplier = 1.05


def downScreenDepth():
    global _screen_depth_multiplier
    _screen_depth_multiplier = 1/1.05

def reverse_time():
    global _time_step_multiplier
    _time_step_multiplier = -1

