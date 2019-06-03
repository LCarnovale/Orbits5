import math
import numpy as np

DELETE_FORCE_LOOP = True
SYSTEM_APPEND_PREALLOC = 100
SYSTEM_PREALLOC_ON_INIT = True

DEFAULT_BUFFER_ATTRS = ['pos', 'vel', 'mass', 'radius', 'active', 'N']
# default_masked_attributes = ['pos', 'vel', 'mass', 'radius']

class SystemError(Exception):
    def __init__(self, msg=None, system=None):
        if not msg:
            msg = 'A System error has occured'
        if system:
            msg += '\nSystem: %s'%system
        super().__init__(msg)

def main():
    pos_ = np.array([
        [0, 0, 0],
        [3, 1, 1],
        [-2, -2, -2],
        [1, 1, 1],
    ])
    mass_ = np.array([
        1,
        2,
        3,
        4,
    ])
    s = System(pos=pos_, mass=mass_)

    return s

class System:
    def __init__(self, 
        prealloc_on_init=SYSTEM_PREALLOC_ON_INIT, 
        **kwargs):
        """
        kwargs:
            Requires atleast one keyword holding an array of values.
            The keys used for keywords become the attribute names used to
            access the supplied arrays.

            Length of the system (what will become System.N) is determined
            by the first seen array. Arrays of length 1 after that will be
            repeated to match the length of the first array.
            
            Each array must have the same first dimension, ie same length,
            but can have different shapes in each element.

            For example, pos can be given as a 5x3 array for 3 particles,
            and mass can be given as a 5x1 array, or shape (5,) array.
        S = System(
            pos=[N x d array of float], <-- Positions of all N particles
            vel=[N x d array of float], <-- Velocities of all N particles
            mass=[N size array of float], <-- Mass of all N particles
            radius=[N size array of float] <-- Radius of all N particles
        )

        Attributes:
            len(self) or self.N : Number of particles, or number of rows
                in all arrays.
            self.Nfull : similar to self.N but ignores the mask, this value
                will never change (unlike self.N) if particles are 'killed',
                unless particles are added or removed.
            self.dim : The dimension of the system. This is assumed to be
                the second element in the shape of the first observed 
                higher-than-1 dimensional array. ie, for a 10x3x... array
                the dim will be stored as 3. This can be set manually if wanted.
            self.active : A mask over all particles in memory that represents
                active particles.

        eg.:
            Colour of i'th particle == info['color'][i]

        Access with:
            >>> S = System( pos=[[1, 1, 1], ...], color=['green', ...],
                                             charge=[-1, 1, ...] )
            >>> S.color[0]
            'green'
            >>> S.charge[1]
            1
        """

        if 'active' in kwargs.keys():
            raise SystemError("Can not have a property called 'active' in a system.")

        ### These must be defined first as they are used in __getattribute__ and get()
        self._mask_enabled = True
        # self._keys = default_masked_attributes.copy()
        self._keys = list(kwargs.keys())
        ### Below here anything can be setted/getted safely ###

        length = 0
        dim = 0
        def fill_vals(key, val):
            if prealloc_on_init:
                shape_ = val.shape
                copy = np.zeros((
                    shape_[0] + SYSTEM_APPEND_PREALLOC, 
                    *shape_[1:]
                ), dtype=val.dtype)
                np.copyto(copy[:len(val)], val)
                val = copy

            super(type(self), self).__setattr__(key, val)

        for k in kwargs:
            val = kwargs[k]
            try:
                val = np.asarray(val)
            except:
                raise SystemError(
                    f"Unable to convert value for {k} into an array"
                )

            try:
                if not dim: dim = len(val[0])
            except:
                pass
            
            len_k = len(val)
            if length:
                if len_k != 1:
                    if len_k != length:
                        raise SystemError(
                            f"Array with inconsistent length given. Expected length {length} got {len_k[0]}"
                        )
                    else:
                        # Set the value
                        try:
                            fill_vals(k, val)
                        except Exception as e:
                            raise SystemError(
                                f"Unable to set value for {k} as an attribute."
                            ) from e
                else:
                    # Repeat the value then set it
                    shape_k = np.shape(val)
                    val = np.tile(val, (length, *(1 for i in shape_k[1:])))
                    # val = np.repeat(val,)
                    fill_vals(k, val)
            else:
                # Set the length of this array as our length
                length = len_k
                fill_vals(k, val)

        

        if length == 0:
            raise SystemError(
                "Must have at least one array supplied when creating a system."
            )
        
        self._head_index = 0
        self._tail_index = len(val)
        self._preallocated = SYSTEM_APPEND_PREALLOC if prealloc_on_init else 0


        # self._active_mask = np.full(length, True)
        fill_vals('_active_mask', np.full(length, True))
        # self.set('_keys_full', [*self._keys, '_active_mask'], raw=True)
        self._keys_full = [*kwargs.keys(), '_active_mask']
        self._dim = dim
        

    def __str__(self):
        s = f'<System: N = {self.N} (+{len(self.active) - self.N} dead), dim = {self.dim}>'
        return s
    
    def __iadd__(self, sys):
        self.append(sys)
        return self
        

    def __getattribute__(self, attr):
        """
        Redirect to the public method get, which behaves
        like __getattribute__ but will use a mask on output 
        when required.
        """
        # masked = super().__getattribute__('_mask_enabled')
        try:
            out = super().__getattribute__('get')(attr)
        except AttributeError:
            raise AttributeError(
                f"{attr} is not an attribute of 'System' nor is it in self.info")
        else:
            return out

    def __getitem__(self, slice_):
        """
        Used to return a new system representing a
        sliced portion of the given system.
        """
        kwargs = {}
        for k in self.names:
            new_val = self.get(k)
            out_shape = new_val.shape
            new_val = new_val[slice_]
            new_val = np.reshape(new_val, (-1, *out_shape[1:]))
            kwargs[k] = new_val
            
        try:
            return System(**kwargs)
        except SystemError:
            return None

    def __setattr__(self, attr, value):
        """Redirect to sys.set(...)"""
        set_f = super().__getattribute__("set")
        return set_f(attr, value)

    def get(self, attr, masked=None, raw=False):
        """
        Used mainly to apply a mask if applicable, otherwise
        handover to the base class's method.

        Used as get('attr'), it is identical to self.attr .
        If masked is provided as True or False, it overrides the
        default masking behaviour.

        raw is False by default, if True, then 
        returns the full allocated array. This is normally useless,
        and is mainly available for use in methods in System.
        If raw is True, masking is ignored completely. 
        """
        if raw:
            return super().__getattribute__(attr)
            
        # Use base classes __getattribute__ for masked attrs and
        # to get the normal output of the attribute:
        _super = super()
        def get_attr(attr):
            return _super.__getattribute__(attr)
        
            
        if masked == None:
            masked = get_attr('_mask_enabled')
            masked_attrs = get_attr('_keys')
            masked = masked and (attr in masked_attrs)

        out = get_attr(attr)
        allocated = get_attr('_keys_full')
        if attr in allocated:
            try:
                out[0]
                head = get_attr('_head_index')
                tail = get_attr('_tail_index')
                out = out[head:tail]
            except:
                pass

        if masked:
            if type(out) == np.ndarray:
                mask = get_attr('get_mask')
                out = out[mask]
        return out

    def set(self, attr, values, index=None, masked=None, raw=False):
        """
        Used to set values with similar functionality to get(name, masked)

        if raw is True (default False) then masked is forced to be False,
        and the raw __setattr__ from super() is used.
        ie,
            >>> sys.set('pos', new_pos, raw=True)
            is equivalent to
            >>> super(type(sys), sys).__setattr__('pos', new_pos)
        This allows all masking to be bypassed and for the actual shape 
        (or even data type) of the array to be changed. It shouldn't need
        to be used outside of the class methods.
        """
        # print(attr, values)
        try:
            if raw:
                super().__setattr__(attr, values)
                return
            if np.any(index == None):
                index_ = slice(0, None)
            else:
                index_ = index

            if masked == None:
                masked = self._mask_enabled
            temp = self.get(attr, masked=False)
            if masked:
                # masked_attrs = self._masked_attributes
                is_masked_attr = attr in self.names
                mask = self.get_mask
                if index == None:
                    if is_masked_attr:
                        temp[mask] = values
                    else:
                        temp = values
                else:
                    if is_masked_attr:
                        temp_ = temp[mask]
                        temp_[index_] = values
                        temp[mask] = temp_
                    else:
                        temp[index_] = values

            else:
                # if attr in self._info:
                #     temp_ = self._info[attr]
                #     temp_[index_] = values
                #     self._info[attr] = temp_
                # else:
                temp[index_] = values
        except AttributeError as e:
            if np.all(index == None):
                super().__setattr__(attr, values)
            else:
                raise e
        else:
            super().__setattr__(attr, temp)

    def append(self, sys):
        """
        Append rows to a system.
        sys: System or dictionary,
            in either case sys.keys() must contain 
            the same values as self.keys()
            The only exception is the key 'active',
            which can be excluded, and will then be assumed
            True for all added rows

            The shape of the values in sys must be identical
            except for the first value, ie: 
                self.get('xxx').shape[1:] == sys.get('xxx').shape[1:]
            The length of the first dimension can change (as this
            value represents the number of rows.)

            The number of rows in sys must be consistent for all keys
        """
        try:
            sys_keys = sys.keys(full=True)
        except:
            sys_keys = list(sys.keys())
        
        self_keys = self.keys(full=True)
        if 'active' not in sys_keys and '_active_mask' not in sys_keys:
            sys_keys.append('_active_mask')
            # We can assume it is a dictionary now,
            # since it doesn't have an 'active' attribute.
            new_active_mask = True
        else:
            new_active_mask = False

        if sorted(self_keys) != sorted(sys_keys):
            raise SystemError(
                'Unable to join systems with different keys.'
            )

        # Get the length
        try:
            n_copy = len(sys.get(sys_keys[0]))
            if n_copy == 0:
                print(f"Given shape: {sys.get(sys_keys[0]).shape}")
                raise SystemError("Got a zero length array in input")
        except Exception as e:
            raise SystemError("Unable to get shape of array to be added") from e
        
        # Double check active values
        if new_active_mask:
            # Again we can assume sys is a dict now
            sys['_active_mask'] = np.full((n_copy), True, dtype=bool)

        # See if we need to preallocate more space
        if n_copy >= self._preallocated:
            print("allocating more")
            # First try and roll back the buffer if possible
            if self._head_index != 0:
                print("rolling")
                old_head = self._head_index
                old_N = self.N
                self._head_index = 0
                for n in self_keys:
                    self.set(n, self.get(n)[old_head:old_head+old_N],  
                             index=slice(0, old_N))
                self._tail_index = old_N
                self._preallocated += old_head

            if n_copy >= self._preallocated:
                print("doing allocation")
                # At this point we know that the head index is 0,
                # so that self.N is essentially the tail index.
                # Now we preallocate rows:
                allocate_new = n_copy + SYSTEM_APPEND_PREALLOC + self.N

                for n in self_keys:
                    old = self.get(n, raw=True)
                    new = np.resize(old, (allocate_new, *old.shape[1:]))
                    self.set(n, new, raw=True)
                # This does not account for the rows that are about to be added 
                self._preallocated = n_copy + SYSTEM_APPEND_PREALLOC
            else:
                print("allocation avoided")
        else:
            print("no allocating")
        print(self._preallocated)
                
        # Now we have enough space to append, do the append.
        # We cannot assume anything about the head or tail
        # index in the code below.
        for n in self_keys:
            old = self.get(n, raw=True)
            print(n, old.shape)
            # try:
            add = sys.get(n)
            add = np.asarray(add)
            if len(add) != n_copy:
                raise SystemError("All values must have consistent lengths to append systems")
            if add.shape[1:] != old.shape[1:]:
                got_shape = list(add.shape)
                wanted_shape = list(old.shape)
                wanted_shape[0] = '*'
                raise SystemError(
f"Unable to append system, got shape {got_shape}, wanted [\
{', '.join((str(x) for x in wanted_shape))}]."
                )
            
            try:
                np.copyto(old[self._tail_index : self._tail_index+n_copy], add)
            except Exception as e:
                print("attribute:", n)
                print("source:", add.shape)
                print("destination:",
                      old[self._tail_index: self._tail_index+n_copy].shape)
                print("tail_index:", self._tail_index)
                print("n_copy:", n_copy)
                raise SystemError("Unable to append systems.") from e

            self.set(n, old, raw=True)

        self._tail_index += n_copy
        self._preallocated -= n_copy


    def new_key(self, key, shape, fill=0.):
        """
        Initialize a new array under the given key.

        shape: the shape of each element in the array,
        eg. give shape=3 for a N x 3 array to fill.

        fill: the scalar value to fill the array with. 
        """
        try:
            new_arr = np.full((self.N, *shape), fill)
        except TypeError:
            new_arr = np.full((self.N, shape), fill)
        

        self.set(key, new_arr, raw=True)
        self.set('_keys', self._keys + [key], raw=True)


    def initialize_info(self, key, dim, fill=0., masked=False):
        """
        Adds an empty N x 'dim' array to self.info under the 'key'

        fill: default 0.0, value to fill the new array with
        masked: default False, if True then output will be masked by the active mask.

        If the key already exists, do nothing.
        """
        raise NotImplementedError("Use 'sys.new_key() instead")


    def add_particle(self, position, velocity=None, mass=None, radius=None):
        """
        Requires a value for all arguments parsed when the object was created,
        including a dict for the new info entry, (the dict that should be
        returned by info[i] for the new i), if info was provided to __init__().
        """
        raise NotImplementedError

    def del_particle(self, i):
        """
        Removes the i'th particle from the system.
        """
        pass

    def kill_particle(self, i):
        """
        Marks the i'th particle as dead, making it ignored by
        the simulation

        """
        # print(f"Killing {i}")
        self._active_mask[self.active_map[i]] = False

    def kill_collisions(self, test, kill_func):
        """
        Requires a test function 'test', and a kill function 'kill_func':
        test(sys, A, B) -> [-1|1]
            Should return 1 if B is to be killed, or -1 if A is to be killed.
            The function will be given the system, and the indexes of A and B.
            *** A and B may be 1D arrays
        kill_func(sys, A, B) -> [-1 | particle index]
            Should make any desired modifications
            (such as changing mass, velocity, etc.) to A, as a result
            a of 'killing' B. ie, A is the survivor of the collision between
            A and B.
            *** If B is to be actually killed / inactivated, the function
                should return B. Deactivating B inside the kill_func can have
                unpredictable (bad) effects. If nothing is being killed, return None

        Returns the number of particles 'deactivated'.
        """
        collisions = self.get_collisions()
        if not collisions.any():
            return 0
        # print(collisions)
        r = test(self, collisions[:, 0], collisions[:, 1])
        mask = np.array([r < 0, r > 0]).transpose()
        # idx_array = collisions[mask]
        A = collisions[np.invert(mask)]
        B = collisions[mask]
        kill_list = np.full(len(A), -1)
        if DELETE_FORCE_LOOP:
            for i in range(len(A)):
                a = A[i:i+1]
                b = B[i:i+1]
                k_res = kill_func(self, a, b)
                if k_res is not None:
                    kill_list[i] = k_res
        else:
            kill_list = kill_func(self, A, B)
        kill_list = np.array(list(set(kill_list)))
        self.kill_particle(kill_list[kill_list >= 0])
        return len(collisions)

    def remove_dead(self):
        """
        Removes particles marked inactive completely
        """
        pass

    def set_prev_pos(self):
        """
        Call this to set the current pos array as the previous pos array,
        use this before applying a change to the array to be able to get the
        change in position.
        """
        self._prev_pos = self._pos.copy()

    def set_pos_delta(self, delta):
        """Used to change the ._pos_delta array."""
        self._pos_delta[self.get_mask] = delta.copy()

    def keys(self, full=False):
        if full:
            return self._keys + ['_active_mask']
        else:
            return self.names

    """
    Properties
    """
    @property
    def names(self):
        return self._keys

    @property
    def active_map(self):
        """
        Returns np.flatnonzero(self._active_mask)
        To allow mapping from a masked index (of an active particle)
        to a true unmasked index
        """
        return np.flatnonzero(self._active_mask)

    @property
    def active(self):
        return self._active_mask
    # def active():
    #     def fget(self):
    #         return self._active_mask
    #     def fset(self, val):
    #         self._active_mask

    # def property()

    @property
    def dead(self):
        # Return a mask of 'dead' particle
        return np.logical_not(self._active_mask)

    @property
    def get_mask(self):
        """
        Returns the self._active_mask if self._mask_enabled is True,
        otherwise returns a self.N length array of True's.
        """
        if self._mask_enabled:
            return self._active_mask
        else:
            return np.full(self.Nfull, True)

    @property
    def N(self):
        return len(self.active[self.active])

    @property
    def Nfull(self):
        # Same as N but always ignores mask
        return len(self.active)

    @property
    def dim(self):
        return self._dim


class ClassicSystem(System):
    def __init__(self, pos, vel=0., mass=0.,
                 radius=0., force=0., angular=0., **kwargs):
        try:
            pos = np.asarray(pos)
        except:
            raise SystemError("Unable to convert given pos into array")
        else:
            p_shape = np.shape(pos)
            if len(p_shape) < 2:
                raise SystemError("Array for pos must be atleast 2D")
            N = p_shape[0]
            # shape_ = p_shape[1:]
            def_scalar_shape = (N, 1)
            def_vector_shape = p_shape

        if np.isscalar(vel):
            vel = np.full(def_vector_shape, vel)
        if np.isscalar(mass):
            mass = np.full(def_scalar_shape, mass)
        if np.isscalar(radius):
            radius = np.full(def_scalar_shape, radius)
        if np.isscalar(force):
            force = np.full(def_vector_shape, force)
        if np.isscalar(angular):
            angular = np.full(def_vector_shape, angular)
    
        # Get N and dim from pos
        super().__init__(**{
            'pos':pos,
            'vel':vel,
            'mass':mass,
            'radius':radius,
            'force':force,
            'angular':angular
        }, **kwargs)

    def get_collisions(self):
        """
        Returns pairs of indexes of particles colliding with eachother
        eg. -> array([ (1,3), (0, 5) ])

        calling kill_particle(np.ravel(check_collision())) will
        'kill' all colliding particles.
        calling kill_particle(check_collision()[:0 or 1]) will
        'kill' one of the colliding particles
        """
        try:
            self.pos
            self.radius
        except AttributeError:
            raise SystemError("collisions requires pos and radius values")
        POS_ALL = np.tile(self.pos, (self.N, 1, 1))
        POS_S = np.tile(self.pos, (1, 1, self.N)).reshape(POS_ALL.shape)
        D = POS_S - POS_ALL  # r
        D = np.linalg.norm(D, 2, axis=-1)
        # D = np.sqrt(np.sum(D**2, axis=-1))
        # Get combined radii:
        RM = self.radius.reshape(1, -1)
        Rgrid = RM + RM.transpose()
        collision_mask = D < Rgrid
        # Make coordinate grid:
        x = np.arange(D.shape[-1])
        y = np.arange(D.shape[-1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        # Collisions:
        row_ = xx[collision_mask]
        col_ = yy[collision_mask]
        row = row_[col_ > row_]
        col = col_[col_ > row_]
        idx_array = np.array([row, col]).transpose()
        return idx_array


    """
    Properties
    """

    @property
    def com(self):
        """Centre of mass"""
        return np.sum(
            self.pos * self.mass.reshape(-1, 1),
            axis=0
        ) / np.sum(self.mass)

if __name__ == '__main__':
    s = main()
