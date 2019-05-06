import numpy as np

PRINT_BUFFER_INLINE = True
default_masked_attributes = None

class Buffer:
    """
    Used to handle the output from sim.buffer(),
    a Buffer object can be indexed and sliced to return
    another buffer with the same original keys but with specific frames.
    """

    def __init__(self, buffer_dict):
        self._dict = buffer_dict
        self._masked_attributes = default_masked_attributes

    def __str__(self):
        if not PRINT_BUFFER_INLINE:
            fields_ = '\n    '.join(['']+[k for k in self._dict]) + "\n"
        else:
            fields_ = ','.join([k for k in self._dict])
        return (
            f"<Buffer length {len(self)}, fields: [{fields_}]>"
        )

    def __add__(self, buffer):
        new = self.copy()
        if buffer == None:
            return self.copy()
        for k in self._dict:
            new._dict[k] = np.append(self._dict[k], buffer[k], axis=0)

        return new

    def __contains__(self, x):
        return self._dict.__contains__(x)

    def __radd__(self, buffer):
        return self.__add__(buffer)

    def __iadd__(self, buffer):
        self = self + buffer

    def __getitem__(self, s):
        if type(s) in [int, slice]:
            out = {}
            try:
                for k in self._dict:
                    val = self._dict[k][s].copy()
                    if type(s) != slice:
                        out[k] = np.array([val])
                    else:
                        out[k] = val
            except IndexError:
                raise BufferError(
                    f"Frame {s} out of bounds for Buffer of length {len(self)}",
                    self
                )
            return Buffer(out)
        elif s in self._dict:
            out = self._dict[s]
            return out
        else:
            raise KeyError

    def __getattr__(self, s):
        return self.__getitem__(s)

    def __len__(self):
        l = len(next(iter(self._dict.values())))
        return l

    def pull(self):
        """
        Return the first frame and delete it.
        """
        if len(self) == 0:
            return None
        f = self[0].copy()
        self._dict = self[1:]._dict
        return f

    def copy(self):
        d = {k: self._dict[k].copy() for k in self._dict}
        return Buffer(d)

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
