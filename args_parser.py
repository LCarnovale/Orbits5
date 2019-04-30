# Used to parse all the arguments for Orbits5
import sys
import time
import numpy as np

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
"-?"  :  [None], # Help.
"-d"  :     [float,	0.025,	True], # Delta time per step
"-n"  :     [int,   20,		True], # Particle count
"-p"  :     [str,   "1",	True], # preset
"-rn" :     [int,   2,      True], # Power of r (F = -GMm/r^n) for preset 4.5
"-rt" :     [str,   False,  False,  True], # Run in real time
"-sp" :     [str,   False,	False,  True], # start paused
"-ss" :     [str,   False,	False,  True], # staggered simulation
"-G"  :     [float, 20,		True], # Gravitational constant
"-pd" :     [str,   False,	False,  True], # Print data
"-sd" :     [float, 2500,	True], # Default screen depth
"-ps" :     [float, 0.01,	True], # Maximum pan speed
"-rs" :     [float, 0.01,	True], # Rotational speed
"-ip" :     [str,   "Earth",True], # Initial pan track
"-ir" :     [str,   "Sun",  True], # Initial rot track
"-sep":     [float, 700,    True], # Separation of bodies A and B in preset 6
"-ae" :     [str,   True,   False,  True], # Auto "exposure"
"-rel":     [str,   False,  False,  True], # Visualise special relativity effects (Experimental)
"-mk" :     [str,   False,	False,  True], # Show marker points
"-ep" :     [int,   360,	True], # Number of points on each ellipse (Irrelevant if SMART_DRAW is on)
"-sf" :     [float, 0.5,	True], # Rate at which the camera follows its target
"-ad" :     [str,   False,	False,  True], # Always draw. Attempts to draw particles even if they are thought not to be on screen
"-vm" :     [float, 150,	True], # Variable mass. Will be used in various places for each preset.
"-vv" :     [float, False,	False,  1], # Draw velocity vectors
"-ds" :     [str,	False,	False,  True], # Draw stars, ie, make the minimum size 1 pixel, regardless of distance.
"-sdp":     [int,   5,		True], # Smart draw parameter, equivalent to the number of pixels per edge on a shape
"-me" :     [int,   400,	True], # Max number of edges drawn
"-ab" :     [int,   False,	False,  20], # Make asteroid belt (Wouldn't recommend on presets other than 3..)
"-es" :     [int,	False,	False,  5], # Make earth satellites
"-WB" :     [str,   False,	False,	True], # Write buffer to file
"-rp" :     [float, False,  False,  0.6], # Make random planets
"-cf" :     [str,	True,  False,  True], # Use complex flares
"-sr" :     [int,   False,  True], # Make rings (really just a thin asteroid belt) around Saturn
"-rg" :     [str,   False,  False,  True], # Record gif shots
"-tn" :     [str,   False,  False,  True], # True n-body simulation. When off program makes some sacrifices for performance boost.
"-asb" :    [int,   4,      True], # Number of bodies in the auto-systems.
"-flim":    [float, False,  True], # Frame limit
"-demo":    [str,   False,  False,  True], # Run demo
"-dfs":     [int,   0,      True], # Draw diffraction spikes (Experimental)
"-df" :     [str, "SolSystem.txt", True], # Path of the data file
"-test":    [str,   False,  False, True], # Test mode
"-getStars":[float, False,  False, 4], # Get stars from the datafile.
"-PM":      [str,   False,  False, True],  # Enter the preset maker
"-dbg":     [str,   False,  False, True],  # Enter debug mode. (Frame by frame with command line options)
"-P?":      [str,   False,  False, True],  # Show available presets and quit
"-AA_OFF":  [str,   True,   False, False]   # Turn off AutoAbort.
}

originalG = args["-G"][1]

if len(sys.argv) > 1:
	if ("-?" in sys.argv):
		# Enter help mode
		print("Welcome to Orbits4T!")

		print("""
This version contains the following presets:
1)  Centre body with '-n' number of planets orbiting in random places. (Default 10)
2)  'Galaxy' kinda thing (Miserable failure, don't waste your time with this one)
3)  THE WHOLE UNIVERSE
4)  Another small test. Large body with a line of small bodies orbiting in a circle.
5)  Repulsive particles on the surface of a sphere, that eventually sort themselves into an even spread. Just cool to watch.
The third one is way better, don't even bother with the others. They were just practice.

Arguments:
Key|Parameter type|Description
   | (if needed)  |
-d :    float       Delta time per step.
-n :    int         Particle count, where applicable.
-p :    string      Preset.
-sp:                Start paused.
-rt:                Start in real time. (Can be toggled during the simulation)
-ss:                Staggered simulation (Hit enter in the terminal to trigger each step)
-G :    float       Gravitational constant.
-pd:                Print debugging data.
-sd:    float       Default screen depth.
-ps:    float       Maximum pan speed.
-rs:    float       Rotational speed.
-ip:    string      Starts the simulation with pan track at a body with the given name. (ONLY PRESET 3)
						The body is found by using the search function, so a HIP id will work too.
-ir:    string      Starts the simulation with rot track at a body with the given name, like -ip. (ONLY PRESET 3)
-mk:                Show marker points (static X, Y, Z and Origin coloured particles)
-ep:    int         Number of points on each ellipse (Irrelevant if SMART_DRAW is on (which it is))
-sf:    float       Rate at which the camera follows its target.
-ad:                (Debug tool) Always draw. Attempts to draw particles even if they are thought not to be on screen
-vm:    float       Variable mass. To be used in relevant places for some presets.
-vv:                Draw velocity and acceleration vectors. Note that while velocity vectors are to scale,
						acceleration vectors are multiplied by 5 when being drawn. (Otherwise they are too short)
						Give a number parameter to scale each vector.
-ds  :              Draw stars, ie, make the minimum size 1 pixel, regardless of distance.
-sdp :  int         Smart draw parameter, equivalent to the number of pixels per edge on a shape.
-me  :  int         Maximum edges, max number of edges drawn on each shape.
-ab  :  int         Make asteroid belt (Wouldn't recommend on presets other than 3..)
-es  :  int         Make earth satellites.
-WB  :              Write buffer to file.
-sr  :  int         Make rings around Saturn, the given number represents how many objects to make.
-dfs :  int         Draw diffraction spikes around stars. Enter the number of spikes after -dfs. *Experimental*
-rp  :  float       Make random planets, give a percentage of stars to have systems.
						(only for preset 3, if stars are also made)
-tn  : (True/False) Runs the simulation in True N-body mode, making calculations of acceleration due the
						gravity of all bodies to all bodies. Much slower but usually much more accurate
						(Really not worth turning on for presets like the solar system)
						If left on, (ie the argument is not used) then the most influencial bodies at the
						start are the only ones that affect that body for the rest of the simulation.
						But, for some presets this is ON by default.
-asb :  int         Number of bodies in auto generated systems.
-demo:              Runs a demo. Only usable in preset 3, goes through bodies looking around them
						then moving onto the next body.
-flim:  float       Frame limit.
-df  :  string      Path of the data file. (Not implemented)
-test:              Enter test mode.* (See below)
-getStars: float	Loads stars from a database of almost 120,000 stars in the night sky. The value
						given with this parameter will be used as the maximum apparent magnitude from
						Earth of the stars loaded. The default is 4.5, which loads about 510 stars.
-PM  :              Enters the preset maker, allowing you to design a preset.
-P?  :              Shows the available presets then exits.
-AA_OFF:            Turn off AutoAbort. (AutoAbort will kill the simulation if two consecutive frames
						last longer than a second, it's only trying to help you not bring your
						computer to a standstill, be careful if you choose to abandon it)
-? : Enter this help screen then exit

Using the program:
  - Use W, A, S, D to move forwards, left, backwards, and right respectively.
  - Use R, F to move up and down respectively.
  - Use the arrow keys to rotate the camera.
  - '[', ']' to decrease and increase delta time.
  - ',', '.' to decrease and increase the screen depth.
  - 'n', 'm' to start recording and playing the buffer. The simulation will be paused while recording.
  - Space to pause the simulation. (Movement is still allowed)
  - 'I' will set the simulation to run at real time (ish).
  - '\\' will reverse time.
  - Click any particle to set the camera to track that particle.
  - Right click any particle to fix the camera's rotation on that particle.
  - Cycle through targeted particles with Tab/shift-Tab. (Available only in preset 3)
		Once a particle is targeted, pressing T and Y will toggle pan and rotational
		tracking respectively.
  - Press 'G' to go to a selected target.
  - To stop tracking, click (and/or right click) on empty space or another particle.
  - To clear the target selection, press C
  - End the simulation with Esc.

*Test mode: There are some hard coded time, position and velocity snapshots for various
bodies in the simulation, with data taken from the same source as the start positions, but
anywhere between 92 minutes and a month later, and so show the correct positions and velocities
that those bodies should have. Test mode will use the delta time step given by the command line
argument (or the default) and nothing else. No graphics will be drawn, instead the program will
simply step its way through to each relevant time until each of the bodies with test data can
have their correct position and velocity compared with the correct values.""")
		exit()
	argv = sys.argv
	for arg in args:
		args[arg].append(False) # This last value keeps track of whether or not the argument has been specified by the user
	for i, arg in enumerate(argv):
		if arg in args:
			if (args[arg][-1]):
				print("%s supplied multiple times." % (arg))
			try:
				if args[arg][2]:
					if argv[i + 1] in args:
						raise IndexError # If the next arg is an arg keyword (eg -p, -d) then the parameter is missing
					args[arg][1] = args[arg][0](argv[i + 1])
				else: # No parameter needed, set it to args[arg][3]
					if (len(argv) > i + 1 and (argv[i + 1] not in args)):
						if (argv[i + 1] == "False"):
							args[arg][1] = False
						elif (argv[i + 1] == "True"):
							args[arg][1] = True
						else:
							args[arg][1] = args[arg][0](argv[i + 1])
					else:
						args[arg][1] = args[arg][3]
				args[arg][-1] = True
			except ValueError:
				print("Wrong usage of {}".format(arg))
			except IndexError:
				print("Missing parameter for {}.".format(argv[i]))

		else:
			if (arg[0] == "-" and arg[1].isalpha()):
				print("Unrecognised argument: '%s'" % (arg))

else:
	print("You haven't used any arguments.")
	print("Either you're being lazy or don't know how to use them.")
	print("For help, run '%s -?'" % (sys.argv[0]))
	time.sleep(1)
	print("Now onto a very lame default simulation...")
	time.sleep(1)

Delta			        = args["-d"][1]
PARTICLE_COUNT	        = args["-n"][1]
preset			        = args["-p"][1]
STAGGERED_SIM	        = args["-ss"][1]
START_PAUSED	        = args["-sp"][1]
PRINT_DATA		        = args["-pd"][1]
defaultScreenDepth	    = args["-sd"][1]
maxPan			        = args["-ps"][1]
rotSpeed		        = args["-rs"][1]
showMarkers		        = args["-mk"][1]
ellipsePoints	        = args["-ep"][1]
smoothFollow	        = args["-sf"][1]
DRAW_VEL_VECS	        = args["-vv"][1]
ALWAYS_DRAW		        = args["-ad"][1]
variableMass	        = args["-vm"][1]
DATA_FILE		        = args["-df"][1]
drawStars		        = args["-ds"][1]
makeAsteroids 	        = args["-ab"][1]
makeSatellites	        = args["-es"][1]
writeBuffer		        = args["-WB"][1]
FRAME_LIMIT 	        = args["-flim"][1]
getStars 		        = args["-getStars"][1]
DEMO                    = args["-demo"][1]

presetMaker             = args["-PM"][1]
presetShow              = args["-P?"][1]

TestMode 				= args["-test"][1]
AUTO_ABORT              = args["-AA_OFF"][1]      # I wouldn't change this unless you know the programs good to go

SMART_DRAW_PARAMETER = args["-sdp"][1]     # Approx number of pixels between each point

MAX_POINTS = args["-me"][1]  # Lazy way of limiting the number of points drawn to stop the program
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
randomPlanets         = args["-rp"][1]
DEFAULT_SYSTEM_SIZE   = args["-asb"][1] # Default number of bodies to add to a system
TRUE_NBODY            = args["-tn"][1]

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
REAL_TIME           = args["-rt"][1]
voidRadius               = 50000000   # Maximum distance of particle from camera
CAMERA_UNTRACK_IF_DIE = True # If the tracked particle dies, the camera stops tracking it
SMART_DRAW = True               # Changes the number of points on each ellipse depeding on distance
FPS_AVG_COUNT = 3        # Frames used to calculate long average. Less->current fps, more->average fps
RECORD_SCREEN = args["-rg"][1]
DRAW_MARKERS = args["-mk"][1]
RELATIVITY_EFFECTS = args["-rel"][1]
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
COMPLEX_FLARE = args["-cf"][1]
SHOW_SCREENDEPTH = True

MAX_RINGS = 80 # Maximum number of rings to draw to create a flare

# Exposure options
EXPOSURE = 1                 # The width of the flares are multiplied by this
# EXP_COEFF = 400
MAX_EXPOSURE = 1e20
AUTO_EXPOSURE = args["-ae"][1]
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
DIFF_SPIKES = args["-dfs"][1]
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
if (preset not in inBuiltPresets or presetShow):
	pass
