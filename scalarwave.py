import numpy as np

class PlaneWaveScalar:

    def __init__(self, k):

        '''
        Class for handling the plane wave traveling
        in the z direction.

        Parameters
        ----------
        k: float or np.ndarray
            Wavenumber in 1/m.
        polarization: list or tuple or np.ndarray
            Polarization expressed in (x,y) plane
        '''

        self._k = k


    def field(self, z):

        '''
        Scalar field at z.
        '''

        return np.exp(1j*self._k * z)
    
class SphericalWaveScalar:

    def __init__(self, k, position, outbound=True):

        '''
        Class for handling an outgoing spherical wave.

        Parameters
        ----------
        k: float or np.ndarray
            Wavenumber (1/m).
        position: list or tuple or np.ndarray
            Position of the source expressed in (x,y,z) coordinates.
        outbound: bool
            If True, the wave is outgoing. If False, the wave is incoming.
        '''

        self._k = k
        self._position = position
        self._outbound = 1 if outbound else -1

    def field(self, x, y, z):

        '''
        Scalar field at x, y, z. Allows broadcasting.
        '''

        x0, y0, z0 = self._position

        r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

        sw = np.exp(1j*self._outbound*self._k*r)/(self._k**r)

        return sw