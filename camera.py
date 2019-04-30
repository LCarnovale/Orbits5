# Calculates positions of particles on screen
import numpy as np
import math

class CameraError(Exception):
    def __init__(self, msg=None):
        if not msg:
            msg = "An error in a Camera object occured"
        super().__init__(msg)

# def vec_rotation(vec, axis, theta):

def rotation_matrix(axis):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    if axis.shape not in [(3,), (3, 1)]:
        raise Exception("Vector rotation only possible in 3D")

    theta = math.sqrt(np.dot(axis, axis))
    if theta == 0:
        return np.eye(3)
    axis = axis / theta # angle is the length of axis
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd , 2 * (bc + ad)     , 2 * (bd - ac)     ],
                     [2 * (bc - ad)     , aa + cc - bb - dd , 2 * (cd + ab)     ],
                     [2 * (bd + ac)     , 2 * (cd - ab)     , aa + dd - bb - cc ]])

class Camera:
    """
    Methods to produce arrays of x, y, minor and major axes, and angle of rotation.
    """
    def __init__(self, sys, pos=None, vel=None, rot=None, look=None,
                screen_depth=20):
        """
        Args:
            sys: System of particles
            pos: sys.dim length array for position
            vel: sys.dim length array for velocity
            rot: sys.dim length array for rotational velocity,
                 where the magnitude represents radians per second,
                 and direction is the axis of rotation.
            look: sys.dim length array for the direction the camera
                 is currently pointin.
        """
        default_Nd = np.zeros(sys.pos[0].shape)

        if sys.dim != 3:
            raise CameraError("This camera class only supports 3D systems.")
        self._sys = sys
        if np.any(pos == None):
            pos = default_Nd.copy()
        if np.any(vel == None):
            vel = default_Nd.copy()
        if np.any(rot == None):
            rot = default_Nd.copy()
        if np.any(look == None):
            look = default_Nd.copy()
            look[0] = 1


        p_shape = pos.shape
        v_shape = vel.shape
        r_shape = rot.shape
        l_shape = look.shape
        sys_shape = sys.pos[0].shape
        if sys_shape != p_shape:
            raise CameraError("Dimension mismatch between sys.pos and given position vector")
        if sys_shape != v_shape:
            raise CameraError("Dimension mismatch between sys.pos and given velocity vector")
        if sys_shape != r_shape:
            raise CameraError("Dimension mismatch between sys.pos and given rotation vector")
        if sys_shape != l_shape:
            raise CameraError("Dimension mismatch between sys.pos and given look vector")

        self._pos = pos
        self._vel = vel
        self._rot = rot; self._prev_rot = rot
        self._screen_depth = screen_depth
        self._look = look

        self._rot_mat = rotation_matrix(self._rot)
        # self._screen_X_axis = np.array()
        self.set_X_Y_axes(new_Y = np.array([0, 0, 1]))
        # print(self._screen_X_axis)
        # print(self._screen_Y_axis)

    def set_X_Y_axes(self, new_X=None, new_Y=None):
        """
        Orientate the screen axes based on the current 'look' vector
        and a given new X or Y vector
        X and Y increase horizontally and vertically towards the top right
        corner of the screen, and are 0 in the centre.
        """
        if np.all(new_X==None) and np.all(new_Y==None):
            raise CameraError("Must provide a new X or Y screen axis to set them")
        elif np.any(new_X) and np.any(new_Y):
            raise CameraError("X and Y axes cannot be passed simultaneously to set_X_Y_axes(), pick one of them.")
        elif np.any(new_X):
            self._screen_X_axis = new_X
            self._screen_Y_axis = np.cross(new_X, self._look)
        elif np.any(new_Y):
            self._screen_Y_axis = new_Y
            self._screen_X_axis = np.cross(self._look, new_Y)
        self._screen_X_axis = self.screen_X_axis_norm
        self._screen_Y_axis = self.screen_Y_axis_norm



    def render(self, buffer=None, frame=0):
        """
        Calculate screen position of particles in a system or a frame
        from a buffer.

        If buffer is given it should be a Buffer object. The first frame
        will be rendered, unless a frame number is given.
        The buffer MUST have at least position, radius and active.

        If no buffer is given, the positions as they exist in self.sys will
        be rendered.

        """
        if buffer:
            frame_data = buffer[frame]
            active = frame_data['active']
            radius = frame_data['radius'][active]
            pos = frame_data['pos'][active]
        else:
            pos = self._sys.pos
            radius = self._sys.radius
            # active = self._sys.get_mask

        N = len(pos)

        # Normalise look vector:
        self._look = self._look / math.sqrt(np.dot(self._look, self._look))
        look_array = np.tile(self._look, (N, 1))

        # Get relative position:
        self_pos = np.tile(self.pos, (N, 1))
        rel_pos  = pos - self_pos
        rel_dist = np.linalg.norm(rel_pos, 2, axis=-1)
        ### Do some math
        # Get distance from camera to point on screen:        \/ row wise dot product
        dist_to_screen    = self._screen_depth * rel_dist / (np.sum(rel_pos*look_array, axis=-1))
        # Vector to point on screen
        rel_pos_on_screen = rel_pos * (dist_to_screen / rel_dist).reshape((N, 1))

        screen_X = np.sum(rel_pos_on_screen * self.screen_X_axis_norm, axis=-1)
        screen_Y = np.sum(rel_pos_on_screen * self.screen_Y_axis_norm, axis=-1)

        # Angular distance (from centre of screen) and width:
        temp_array = self._screen_depth / dist_to_screen
        temp_array[temp_array>1]  =  1.0
        temp_array[temp_array<-1] = -1.0
        angle_centre = np.arccos(temp_array)
        angle_width  = np.arcsin(radius / rel_dist)

        # Distance from point on screen to screen centre
        dist_screen_centre = np.sqrt(screen_X**2 + screen_Y**2)
        major_axis = 2 * (dist_screen_centre +
                     self._screen_depth * np.tan(angle_width - np.arctan(dist_screen_centre / self._screen_depth)))
        minor_axis = 2 * self._screen_depth * np.tan(angle_width)

        # Get rotation of oval on screen:
        rot = np.zeros(len(screen_Y))
        non_z_mask = screen_X != 0
        rot[non_z_mask] = np.arctan(screen_Y[non_z_mask] / screen_X[non_z_mask])
        # rot[screen_X == 0] = 0
        # Correct for angle domain problems
        # (ie 3rd quadrant angle should be negative etc.)
        neg_x = screen_X < 0
        neg_y = screen_Y < 0
        pos_y = np.invert(neg_y)
        rot[np.logical_and(neg_x, neg_y)] -= np.pi
        rot[np.logical_and(neg_x, pos_y)] += np.pi

        return (screen_X, screen_Y, major_axis, minor_axis, rot)



    def step(self, t_step):
        self._pos  += t_step * self._vel
        if np.any(self._rot):
            angle = np.linalg.norm(self._rot, 2)
            if np.any(self._rot != self._prev_rot):
                self._prev_rot = self._rot.copy()
                self._rot_mat = rotation_matrix(self._rot)

            self._rot = np.dot(self._rot_mat, self._rot)


    def look():
        doc = "Direction the camera is pointing"
        def fget(self):
            return self._look
        def fset(self, value):
            self._look = value
        def fdel(self):
            del self._look
        return locals()
    look = property(**look())

    def pos():
        doc = "The pos property."
        def fget(self):
            return self._pos
        def fset(self, value):
            self._pos = value
        def fdel(self):
            del self._pos
        return locals()
    pos = property(**pos())

    @property
    def screen_X_axis_norm(self):
        return self._screen_X_axis / math.sqrt(np.dot(self._screen_X_axis, self._screen_X_axis))
    @property
    def screen_Y_axis_norm(self):
        return self._screen_Y_axis / math.sqrt(np.dot(self._screen_Y_axis, self._screen_Y_axis))

    @property
    def sys(self):
        return self._sys

    @property
    def mass(self):
        # Perhaps mass could be useful to simulate falling under gravity?
        return 1
