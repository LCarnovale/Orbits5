# Used to parse all the arguments for Orbits5
import sys
import time
import numpy as np

def get_arg_val(arg, default=None):
	"""
	Get the value of a supplied argument.
	arg: key for the argument
	default: [optional] if the argument was given by the use,
			 return this instead of the normal default.
	"""
	if default == None or args[arg][0][-1]:
		return args[arg][0][1]
	else:
		return default

def arg_supplied(arg):
	return args[arg][-1]


#############################################################################################################################
# args:                                                                                                                     #
# <key> : [<type>, <default value>, <requires parameter>, [<second default value>], [changed]]                              #
# second default value is only necessary if <requires parameter> is true.                                                   #
# if true, then the algorithm looks for a value after the key.                                                              #
# if false, the algorithm still looks for a value after the key but if no value is given the second default value is used.  #
# The final value indicates if the argument has been supplied, to know if the user has specified a value                    #
# 	 (Useful if the default varies depending on other arguments)                                                            #
# NOTE: THIS LAST VALUE WILL BE ADDED AUTOMATICALLY, DO NOT INCLUDE IT IN THE CODE                                          #
# ANOTHER NOTE: A key is defined as anything starting with a dash '-' then a letter. A dash followed by a number would be   #
#    read as a negative number.                                                                                                     #
#                                                                                                                           #
########### PUT DEFAULTS HERE ###############################################################################################
args = {#   [<type>   \/	 <Req.Pmtr>  <Def.Pmtr>
# "-?"  :  [None], # Help.
"-d"  :     ([float,	0.025,	True], "Delta time per step"),
"-n"  :     ([int,   20,		True], "Particle count"),
"-p"  :     ([str,   "1",	 True], "preset"),
"-noc":     ([str,   True,   False,  False], "Disable collision checking."),
"-bp" :     ([str,   False,  False,  True], "Buffer while paused."),
"-mb" :     ([int,   5000,   True], "Maximum number of frames to buffer"),
"-dr" :     ([float, 15,     True], "Disc radius (for generic usage)"),
"-bs" :     ([int,   1,      True],   "Steps to perform per step when buffering."),
"-sm" :     ([float, 1,      True], "Speed multiplier for all particles (at the start)"),
"-ke" :     ([float, 1,      True], "Electric force constant"),
"-kb" :     ([float, 1,      True], "Magnetic force constant"),
"-bounce":  ([str,   False,  False, True], "Bounce, doo doo do do do do do, dOO"),
"-bdamp":   ([float, 1.,     True], "Damping of bouncing, 1->no damping, 0->total damping"),
"-vdamp":   ([float, 1.,     True], "Damping of velocity with each step. 1 is no damping."),
"-spin":    ([str,   False,  False, True], "Model spin of particles"),
"-kick":    ([float, 40,     True], "Magnitude of 'kick' given to an\nobject or system, where applicable."),
"-db" :     ([str,   False,   False, True], "Dark buffer, ie don't draw while buffering."),
"-mi" :     ([str,   "leapfrog", True], "Method of integration"),
"-rn" :     ([int,   2,      True], "Power of r (F = -GMm/r^n) for preset 4.5"),
"-rt" :     ([str,   False,  False,  True], "Run in real time"),
"-sp" :     ([str,   False,	False,  True], "start paused"),
"-ss" :     ([str,   False,	False,  True], "staggered simulation"),
"-G"  :     ([float, 15,	True], "Gravitational constant"),
"-pd" :     ([str,   False,	False,  True], "Print data"),
"-sd" :     ([float, 2500,	True], "Default screen depth"),
"-ps" :     ([float, 0.01,	True], "Maximum pan speed"),
"-rs" :     ([float, 0.01,	True], "Rotational speed"),
"-ip" :     ([str,   "Earth",True], "Initial pan track"),
"-ir" :     ([str,   "Sun",  True], "Initial rot track"),
"-sep":     ([float, 700,    True], "Separation of bodies A and B in preset 6"),
"-ae" :     ([str,   True,   False,  True], "Auto 'exposure'"),
"-rel":     ([str,   False,  False,  True], "Visualise special relativity effects (Experimental)"),
"-mk" :     ([str,   False,	False,  True], "Show marker points"),
"-ep" :     ([int,   360,	True], "Number of points on each ellipse (Irrelevant if SMART_DRAW is on)"),
"-sf" :     ([float, 0.5,	True], "Rate at which the camera follows its target"),
"-ad" :     ([str,   False,	False,  True], "Always draw. Attempts to draw particles \neven if they are thought not to be on screen"),
"-vm" :     ([float, 150,	True], "Variable mass. Will be used in various \nplaces for each preset."),
"-vv" :     ([float, False,	False,  1], "Draw velocity vectors"),
"-ds" :     ([str,	False,	False,  True], "Draw stars, ie, make the minimum size 1 pixel, \nregardless of distance."),
"-sdp":     ([int,   5,		True], "Smart draw parameter, equivalent to the \nnumber of pixels per edge on a shape"),
"-me" :     ([int,   80,	True], "Max number of edges drawn"),
# "-ab" :     ([int,   False,	False,  20], "Make asteroid belt (Wouldn't recommend on presets other than 3..)"),
"-es" :     ([int,	False,	False,  5], "Make earth satellites"),
"-WB" :     ([str,   False,	False,	True], "Write buffer to file"),
"-rp" :     ([float, False,  False,  0.6], "Make random planets"),
"-cf" :     ([str,	True,  False,  True], "Use complex flares"),
"-sr" :     ([int,   False,  True], "Make rings (really just a thin asteroid belt) around Saturn"),
"-rg" :     ([str,   False,  False,  True], "Record gif shots"),
"-tn" :     ([str,   False,  False,  True], "True n-body simulation. When off program makes \nsome sacrifices for performance boost."),
"-asb" :    ([int,   4,      True], "Number of bodies in the auto-systems."),
"-flim":    ([float, False,  True], "Frame limit"),
"-demo":    ([str,   False,  False,  True], "Run demo"),
"-dfs":     ([int,   0,      True], "Draw diffraction spikes (Experimental)"),
# "-df" :     ([str, "SolSystem.txt", True], "Path of the data file"),
"-test":    ([str,   False,  False, True], "Test mode"),
"-getStars":([float, False,  False, 4], "Get stars from the datafile."),
"-PM":      ([str,   False,  False, True], "Enter the preset maker"),
"-dbg":     ([str,   False,  False, True], "Enter debug mode. (Frame by frame with command line options)"),
"-P?":      ([str,   False,  False, True], "Show available presets and quit"),
"-AA_OFF":  ([str,   True,   False, False], "Turn off AutoAbort."),
}

# originalG = args["-G"][1]

if len(sys.argv) > 1:
	if ("-?" in sys.argv):
		# Enter help mode
		print("Welcome to Orbits4T!")
		arglist = [(arg, args[arg][0], args[arg][1]) for arg in args]
		arglist_str = []
		get_str_from_type = lambda x: str(x).split(' ')[1][1:-2]
		for key, props, desc in arglist:
			arg_t = get_str_from_type(props[0])
			dflt = props[1]
			p_req = str(props[2])
			p_dflt = ('' if props[2] else str(props[3]))
			s = f"{key:.<9} : {arg_t:.<5} : {dflt:.<10} : {p_req:.<6} : {p_dflt:.<10}"
			if desc: 
				s_len = len(s)
				s += " : "
				if '\n' in desc:
					lines = desc.split('\n')
					s += lines[0]
					for l in lines[1:]:
						s += f"\n.{' ':<{s_len}}. {l}"
				else:
					s += desc

			arglist_str.append(s)
		arglist_full = '\n'.join(arglist_str)

		print(f"""
Orbits5 - Dynamic n-body simulator

For arguments where 'Value needed' is true, then a value must be supplied after the key
in the command line arguments. If a value is not needed, then the parameter is by
default set to 'Default value' if a value is not supplied.


-? : Enter this help screen then exit

Using the program:
  - Use W, A, S, D to move forwards, left, backwards, and right respectively.
  - Use R, F to move up and down respectively.
  - Use the arrow keys to rotate the camera.
  - Space to pause the simulation. (Movement is still allowed)
  - End the simulation with Esc.

Arguments:
{' Key':<9} | {'Type':<5} | {'Default':<10} | {'Value': <6} | {'Default':<10}
{'   ':<9} | {'    ':<5} | {'       ':<10} | {'needed':<6} | {' value':<10}
{arglist_full}
 
""")
# Stuff from Orbits4T:
# 
# 
#   - '[', ']' to decrease and increase delta time.
#   - ',', '.' to decrease and increase the screen depth.
#   - 'n', 'm' to start recording and playing the buffer. The simulation will be paused while recording.
#   - 'I' will set the simulation to run at real time (ish).
#   - '\\' will reverse time.
#   - Click any particle to set the camera to track that particle.
#   - Right click any particle to fix the camera's rotation on that particle.
#   - Cycle through targeted particles with Tab/shift-Tab. (Available only in preset 3)
# 		Once a particle is targeted, pressing T and Y will toggle pan and rotational
# 		tracking respectively.
#   - Press 'G' to go to a selected target.
#   - To stop tracking, click (and/or right click) on empty space or another particle.
#   - To clear the target selection, press C

# *Test mode: There are some hard coded time, position and velocity snapshots for various
# bodies in the simulation, with data taken from the same source as the start positions, but
# anywhere between 92 minutes and a month later, and so show the correct positions and velocities
# that those bodies should have. Test mode will use the delta time step given by the command line
# argument (or the default) and nothing else. No graphics will be drawn, instead the program will
# simply step its way through to each relevant time until each of the bodies with test data can
# have their correct position and velocity compared with the correct values.
		exit()
	argv = sys.argv
	for arg in args:
		args[arg][0].append(False) # This last value keeps track of whether or not the argument has been specified by the user
	for i, arg in enumerate(argv):
		if arg in args:
			if (args[arg][0][-1]):
				print("%s supplied multiple times." % (arg))
			try:
				if args[arg][0][2]:
					if argv[i + 1] in args:
						raise IndexError # If the next arg is an arg keyword (eg -p, -d) then the parameter is missing
					args[arg][0][1] = args[arg][0][0](argv[i + 1])
				else: # No parameter needed, set it to args[arg][0][3]
					if (len(argv) > i + 1 and (argv[i + 1] not in args)):
						if (argv[i + 1] == "False"):
							args[arg][0][1] = False
						elif (argv[i + 1] == "True"):
							args[arg][0][1] = True
						else:
							args[arg][0][1] = args[arg][0][0](argv[i + 1])
					else:
						args[arg][0][1] = args[arg][0][3]
				args[arg][0][-1] = True
			except ValueError:
				print("Wrong usage of {}".format(arg))
			except IndexError:
				print("Missing parameter for {}.".format(argv[i]))

		else:
			if (arg[0] == "-" and arg[1].isalpha()):
				print("Unrecognised argument: '%s'" % (arg))

# else:
# 	print("You haven't used any arguments.")
# 	print("Either you're being lazy or don't know how to use them.")
# 	print("For help, run '%s -?'" % (sys.argv[0]))
# 	time.sleep(1)
# 	print("Now onto a very lame default simulation...")
# 	time.sleep(1)

Delta			        = get_arg_val("-d")
PARTICLE_COUNT	        = get_arg_val("-n")
PRESET			        = get_arg_val("-p")
STAGGERED_SIM	        = get_arg_val("-ss")
START_PAUSED	        = get_arg_val("-sp")
# PRINT_DATA		        = get_arg_val("-pd")
defaultScreenDepth	    = get_arg_val("-sd")
maxPan			        = get_arg_val("-ps")
rotSpeed		        = get_arg_val("-rs")
showMarkers		        = get_arg_val("-mk")
ellipsePoints	        = get_arg_val("-ep")
smoothFollow	        = get_arg_val("-sf")
DRAW_VEL_VECS	        = get_arg_val("-vv")
ALWAYS_DRAW		        = get_arg_val("-ad")
variableMass	        = get_arg_val("-vm")
# DATA_FILE		        = get_arg_val("-df")
drawStars		        = get_arg_val("-ds")
# makeAsteroids 	        = get_arg_val("-ab")
# makeSatellites	        = get_arg_val("-es")
# writeBuffer		        = get_arg_val("-WB")
FRAME_LIMIT 	        = get_arg_val("-flim")
getStars 		        = get_arg_val("-getStars")
# DEMO                    = get_arg_val("-demo")

presetMaker             = get_arg_val("-PM")
presetShow              = get_arg_val("-P?")

TestMode 				= get_arg_val("-test")
AUTO_ABORT              = get_arg_val("-AA_OFF")      # I wouldn't change this unless you know the programs good to go

SMART_DRAW_PARAMETER = get_arg_val("-sdp")     # Approx number of pixels between each point

MAX_POINTS = get_arg_val("-me")  # Lazy way of limiting the number of points drawn to stop the program
							 # grinding to a halt everytime you get too close to a particle



# Time lengths constants
MINUTE  = 60
HOUR    = 60 *	MINUTE
DAY     = 24 *	HOUR
YEAR    = 365 * DAY

# Distance constants
LIGHT_SPEED = 299792458
LIGHT_YEAR  = LIGHT_SPEED * YEAR
AU      = 149597870700
PARSEC  = 3.085677581e+16

# Preset 2
PRESET_2_MIN_DIST = 25
PRESET_2_MAX_DIST = 150

# Preset 2.5
PRESET_2p5_CUBE_WIDTH = 400

# Preset 3
AsteroidsStart 	 = 249.23 * 10**9 # Places the belt roughly between Mars and Jupiter.
AsteroidsEnd 	 = 740.52 * 10**9 # Couldn't find the actual boundaries (They're probably pretty fuzzy)
AsteroidsMinMass = 0.0001 * 10**15
AsteroidsMaxMass = 1	  * 10**23
AsteroidsDensity = 1500
  # Saturn ring constants
STRN_RING_MIN_MASS    = 10
STRN_RING_MAX_MASS    = 1000
STRN_RING_DENSITY     = 1000
STRN_RING_MIN_RADIUS  = 7e6
STRN_RING_MAX_RADIUS  = 50e6
STRN_RING_THICKNESS   = 1e5
  # Random planets settings
randomPlanets         = get_arg_val("-rp")
DEFAULT_SYSTEM_SIZE   = get_arg_val("-asb") # Default number of bodies to add to a system
TRUE_NBODY            = get_arg_val("-tn")

# Preset 4
PRESET_4_MIN_RADIUS = 40
PRESET_4_MAX_RADIUS = 400

# Physical constants
REAL_G       = 6.67408e-11
EARTH_MASS   = 5.97e24   # These are all the actual values, not simulated.
EARTH_RADIUS = 6.371e6
SUN_MASS     = 1.989e30
SUN_RADIUS   = 695.7e6
SOL_LUM      = 3.827e+26 # Watts

# Random Planet Settings
MIN_PERIOD = 20  * DAY
MAX_PERIOD = 150 * YEAR
MAX_PLANET_COUNT = 12
MIN_PLANET_COUNT = 1

# Buffer constants
BUFFER_NORMAL = 0
BUFFER_RECORD = 1
BUFFER_PLAY = 2
BUFFER_DONT_DRAW = 9


# Simulation settings
REAL_TIME           = get_arg_val("-rt")
voidRadius               = 50000000   # Maximum distance of particle from camera
CAMERA_UNTRACK_IF_DIE = True # If the tracked particle dies, the camera stops tracking it
SMART_DRAW = True               # Changes the number of points on each ellipse depeding on distance
FPS_AVG_COUNT = 3        # Frames used to calculate long average. Less->current fps, more->average fps
RECORD_SCREEN = get_arg_val("-rg")
DRAW_MARKERS = get_arg_val("-mk")
RELATIVITY_EFFECTS = get_arg_val("-rel")
RELATIVITY_SPEED = 0

SCREEN_SETUP  = False        # True when the screen is made, to avoid setting it up multiple times
LOW_FPS_LIMIT = 0.25         # If the frame rate is below this for two consecutive frames then abort the sim
PAUSE_ON      = 1
PAUSE_OFF     = -1           # setting MainLoop.pause to one of these will pause or unpause the simulation

MAX_VISIBILE_MAG = 15
# Camera constants
DEFAULT_ROTATE_FOLLOW_RATE = 0.04
GOTO_DURATION       = 6     # Approximately the number of seconds taken to travel to an object using pan track toggling
AUTO_RATE_CONSTANT  = 10    # A mysterious constant which determines the autoRate speed, 100 works well.
FOLLOW_RATE_COEFF   = 0.4
FOLLOW_RATE_BASE    = 1.1
TRAVEL_STEPS_MIN    = 100   # Number of steps to spend flying to a target (at full speed, doesn't include speeding up or slowing down)
MAX_DRAW_DIST       = 3 * LIGHT_YEAR  # Maximum distance to draw bodies that don't have a given magnitude (ie planets, stars are not affected)
MAX_PAN_DIST        = 100   # When toggling to track a target, if the camera is more than this many radii of the planet then
							# the camera will be moved to be within this distance of the target.

DEFAULT_ZERO_VEC = [0, 0, 0]
DEFAULT_UNIT_VEC = [1, 0, 0]

# Drawing/Visual constants
MAG_SHIFT  = 0               # All apparent magnitudes are shifted by this amount before being drawn
MIN_CLICK_RESPONSE_SIZE = 15 # Radius of area (in pixels) around centre of object that can be clicked
							 # depending on its size
MIN_BOX_WIDTH = 50
COMPLEX_FLARE = get_arg_val("-cf")
SHOW_SCREENDEPTH = True

MAX_RINGS = 80 # Maximum number of rings to draw to create a flare

# Exposure options
EXPOSURE = 1                 # The width of the flares are multiplied by this
# EXP_COEFF = 400
MAX_EXPOSURE = 1e20
AUTO_EXPOSURE = get_arg_val("-ae")
AUTO_EXPOSURE_STEP_UP = 0.2      # Coeffecient of the log of the current exposure to step up and down
AUTO_EXPOSURE_STEP_DN = 0.9     # When decreasing exposure, mimic a 'shock' reaction
#                               # By going faster down in exposure. (This is close the human eye's behaviour i think)

REFERENCE_INTENSITY = 4e-7     # 'R', The intensity of a zero magnitude object. Theoretically in units of W/m^2
INTENSITY_BASE = 10**(5/2)      # 'C', Intensity of a body = R*C^( - apparent magnitude)
MIN_VISIBLE_INTENSITY = 4E-70   # Only compared with the intensity after exposure has been applied.
  # Auto Exposure Profile:
REFERENCE_EXPOSURE = 1e6        # Exposure when looking at a zero-mag object
EXPOSURE_POWER = 1.3            # AutoExposure = REF_EXP * e^(EXP_PWR*Magnitude)
AUTO_EXP_BASE = np.exp(EXPOSURE_POWER)

# Flare options
# FLARE_RAD_EXP = 1.5
FLARE_BASE = 5e5             # Scales the size of the flares
FLARE_POLY_POINTS = 20
FLARE_POLY_MAX_POINTS = 100
MIN_RMAG = 0.01             # min 'brightness' of the rings in a flare. Might need to differ across machines.
  # Diffraction variables
DIFF_SPIKES = get_arg_val("-dfs")
PRIMARY_WAVELENGTH = 600E-9  # Average wavelength of light from stars. in m.
FOCAL_LENGTH = 5e-3           # Focal length of the 'eyeball' in the camera
PUPIL_WIDTH_FACTOR = 0.01   # This multiplied by the exposure gives the width of the pupil in calculations
AIRY_RATIO = 1/10            # The ratio of the intensity between successive maximums in an airy disk
  # Not user defined, don't change these. Computed now to save time later
AIRY_COEFF = 1 / np.log(AIRY_RATIO)
DIFFRACTION_RADIUS = FLARE_BASE * FOCAL_LENGTH * PRIMARY_WAVELENGTH / ( PUPIL_WIDTH_FACTOR )
# DIFFRACTION_RADIUS is the radius of the first order light ring,
# with a brightness of AIRY_RATIO * the centre brightness, and so on for each ring
# The flare will extend for all the rings that are above the minimum light intensity

# COMFORTABLE_INTENSITY
currentDisplay = ""          # Holds the last data shown on screen incase of an auto-abort
lowestApparentMag = None     # The lowest apparent mag of an object on screen, used for auto exposure
totalIncidentIntensity = 0   # The total intensity of light falling on the camera in each frame

fullFlareBuffer = []
## Load presets
inBuiltPresets = ["1", "2", "3", "4", "5"]
# if (preset not in inBuiltPresets or presetShow):
# 	pass
