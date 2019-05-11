import numpy as np


PRINT_BUFFER_INLINE = True
BUFFER_APPEND_PREALLOC = 100 # Number of rows to preallocate when adding to a buffer
default_masked_attributes = None


class BufferError(Exception):
    def __init__(self, msg=None, buffer=None):
        if not msg:
            msg = 'A Buffer error has occured'
        if buffer:
            msg += f'\nBuffer: {buffer}'

        super().__init__(msg)
class Buffer:
    """
    Used to handle the output from sim.buffer(),
    a Buffer object can be indexed and sliced to return
    another buffer with the same original keys but with specific frames.
    """

    def __init__(self, buffer_dict):
        self._dict = buffer_dict
        self._masked_attributes = default_masked_attributes
        # End of the used/initialized data, while _head_index == 0, 
        # this is essentially the external size of the buffer, 
        # what the outside world sees as len(self)
        self._tail_index = (next(iter(buffer_dict.values()))).shape[0]
        # The number of rows in memory taken up by the buffer 
        # (some may be empty/uninitialized)
        self._allocated_size = 1*self._tail_index
        # The starting index of the array. Quicker to change the 
        # head position after pulling instead of rolling the 
        # whole array back each time (For large buffers)
        # len(self) is just self._tail_index - self._head_index
        self._head_index = 0

    def __str__(self):
        if not PRINT_BUFFER_INLINE:
            fields_ = '\n    '.join(['']+[k for k in self._dict]) + "\n"
        else:
            fields_ = ','.join([k for k in self._dict])
        fields_ = f"{len(self._dict)} fields" 
        return (
            f"<Buffer length {len(self)}, fields: [{fields_}]>"
        )

    def __add__(self, buffer):
        ### Performs addition in-place ###
        if buffer == None:
            return self
        # new = self.copy()
        # Get the number of rows to be added
        new_rows = len(buffer)
        # Compare with available preallocated rows
        avail_rows = self._allocated_size - self._tail_index
        if new_rows < avail_rows:
            # Go ahead with copy
            pass
        else:
            # First try and roll back the buffer if possible
            if self._head_index != 0:
                for k in self._dict:
                    self._dict[k] = np.roll(self._dict[k], -self._head_index, axis=0)
                avail_rows += self._head_index
                self._tail_index -= self._head_index
                self._head_index = 0
            # Check if we still need more space 
            if new_rows > avail_rows:
                # We still need to allocate more space to the new buffer.
                allocate = self._allocated_size + new_rows - avail_rows + BUFFER_APPEND_PREALLOC
                for k in self._dict:
                    self._dict[k] = np.resize(self._dict[k], (allocate, *self._dict[k].shape[1:]))
                self._allocated_size = allocate
        # By now we can safely do the copy
        self._tail_index += new_rows
        for k in self._dict:
            try:
                np.copyto(
                    self._dict[k][self._tail_index - new_rows : self._tail_index ],
                    buffer[k][:len(buffer) ]
                )
                
            except Exception as e:
                raise BufferError(f"""Unable to append buffer key {k},
Unable to append {buffer[k].shape} to {self._dict[k].shape}""") from e
                

        return self

    def __contains__(self, x):
        return self._dict.__contains__(x)

    def __radd__(self, buffer):
        return self.__add__(buffer)

    def __iadd__(self, buffer):
        self = self + buffer

    def __getitem__(self, s):
        try:
            if s >= len(self):
                raise BufferError(
                    f"Frame {s} out of bounds for Buffer of length {len(self)}"
                )
        except:
            pass
        # s must be either a string, integer or slice.
        try:
            mask = np.arange(len(self)) + self._head_index
            s = mask[s]
        except:
            # s must now be a string
            if s in self._dict:
                out = self._dict[s]
                return out
            else:
                # 's' does not exist
                raise AttributeError
        else:
            # s is now an nd-array or int
            out = {}
            try:
                for k in self._dict:
                    pre = self._dict[k]
                    pre_shape = np.shape(pre)

                    val = pre[s]
                    val_shape = np.shape(val)
                    if not val_shape or len(val_shape) != len(pre_shape):
                        # 'out' will have had a dimension shaved off, 
                        # if the index is a single int. add it back on.
                        out[k] = np.array([val])
                    else:
                        out[k] = val
            except IndexError:
                print(f"k: {k}")
                print(f"dict shape: {self._dict[k].shape}")
                raise BufferError(
                    f"Frame {s-len(self)} out of bounds for Buffer of length {len(self)}"
                )
            return Buffer(out)
        # else:
        #     raise KeyError

    def __getattr__(self, s):
        return self.__getitem__(s)

    def __len__(self):
        l = self._tail_index - self._head_index
        return l

    def pull(self):
        """
        Return the first frame and delete it.
        """
        if len(self) == 0:
            return None
        f = self[0].copy()
        self._head_index += 1
        # self._dict = self[1:]._dict
        # for k in self._dict:
        #     selfEnd of the used/initialized data, while _head_index == 0, 
        # this is essentially the _dict[k] = np.roll(
        # self._dict[k], -1, axis=0) as len(self)
        # self._tail_index -= 1
        return f

    def copy(self):
        d = {k: self._dict[k].copy() for k in self._dict}
        return Buffer(d)

# len(self) is just self._tail_index - self._head_index
    @property
    def size(self):
        return len(self)

    @property
    def keys(self):
        return self._dict.keys()

# class Window(Buffer):
#     def __init__(self, Buffer):
#         """
#         Identical appearance to a buffer, but is instantiated
#         using an existing buffer, and acts as a pointer or 'window'
#         for the given buffer.
