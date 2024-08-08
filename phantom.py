from typing import Union, List, Tuple
import numpy as np

class _Phantom:

    def __init__(self, pixelsize: float, shape: Union[List[int], Tuple[int]], 
                 background=1.0, *args, **kwargs):

        '''
        Base class for phantom objects.
        '''

        self._background = background
        self._amplitude = np.ones(shape) * background
        self._phase = np.zeros(shape)

        self._pixelsize = pixelsize
        self._shape = shape

        self._x = pixelsize * np.arange(-(shape[1]-1)/2, (shape[1]-1)/2+1)
        self._y = pixelsize * np.arange(-(shape[0]-1)/2, (shape[0]-1)/2+1)
        self._extent = (self._x.min(), self._x.max(), self._y.max(), self._y.min())
        self._x, self._y = np.meshgrid(self._x, self._y)

        self._args = args
        self._kwargs = kwargs
    
    def __mul__(self, other: '_Phantom'):

        '''
        Multiplies two phantoms together.
        '''

        if type(self) != type(other):
            raise TypeError('Both objects must be instances of the same shape.')

        if self._pixelsize != other._pixelsize:
            raise ValueError('Pixelsizes must be equal.')

        if self._shape != other._shape:
            raise ValueError('Shapes must be equal.')
        
        result = type(self)(
            self._pixelsize,
            self._shape,
            background=self._background,
            *self._args,
            **self._kwargs
        )

        result.setamplitude(self.amplitude * other.amplitude)
        result.setphase(self.phase + other.phase)

        return result
    
    def refine(self, factor: int):

        '''
        Refines the phantom by a factor.

        Parameters
        ----------
        factor: int
            Refinement factor.
        '''

        return self.__class__(
            self._pixelsize/factor,
            factor * (np.array(self._shape)-1),
            background=self._background,
            *self._args,
            **self._kwargs
        )
    
    def getpx(self):
        return self._pixelsize

    def getshape(self):
        return self._shape
    
    def getamplitude(self):
        return self._amplitude
    
    def setamplitude(self, amplitude: np.ndarray):

        '''
        Sets the amplitude of the phantom.

        Parameters
        ----------
        amplitude: np.ndarray
            Amplitude of the phantom.
        '''

        self._amplitude = amplitude

    def getphase(self):
        return self._phase
    
    def setphase(self, phase: np.ndarray):

        '''
        Sets the phase of the phantom.

        Parameters
        ----------
        phase: np.ndarray
            Phase of the phantom.
        '''

        self._phase = phase

    def getfield(self):
        return self._amplitude * np.exp(1j*self._phase)
    
    def getextent(self):
        return self._extent
    
    def getx(self):
        return self._x[0,:]
    
    def gety(self):
        return self._y[:,0]
    
    pixelsize = property(getpx, None, None, 'Pixelsize of the phantom.')
    shape = property(getshape, None, None, 'Shape of the phantom.')
    amplitude = property(getamplitude, setamplitude, None, 'Amplitude of the phantom.')
    phase = property(getphase, setphase, None, 'Phase of the phantom.')
    field = property(getfield, None, None, 'Field of the phantom.')
    extent = property(getextent, None, None, 'Extent of the phantom (min(x), max(x), max(y), min(y)).')
    x = property(getx, None, None, 'x-axis of the phantom.')
    y = property(gety, None, None, 'y-axis of the phantom.')
    

class Circle(_Phantom):

    def __init__(self, pixelsize: float, shape: Union[List[int], Tuple[int]], 
                 radii: Union[List[float], Tuple[float]], centers: Union[List[float], Tuple[float]],
                 amplitudes: Union[List[float], Tuple[float]], phases: Union[List[float], Tuple[float]],
                 background=1.0):

        '''
        Circle phantom.

        Parameters
        ----------
        pixelsize: float
            Pixelsize of the phantom.
        shape: List or Tuple
            Shape of the returned phantom field.
        radii: List or Tuple
            List of radii of the circles.
        centers: List or Tuple
            List of centers of the circles.
        amplitudes: List or Tuple
            List of amplitudes of the circles.
        phases: List or Tuple
            List of phases of the circles.
        background: float
            Background amplitude of the phantom. Default is 1.0.
        '''

        super(Circle, self).__init__(
            pixelsize, shape, background, radii, centers, amplitudes, phases
        )

        self._radii = radii
        self._centers = centers
        self._amplitudes = amplitudes
        self._phases = phases
        self._pixelsize = pixelsize
        self._shape = shape

        self.amplitude = self._prep_amplitude()
        self.phase = self._prep_phase()

    def _prep_amplitude(self):

        '''
        Prepares the amplitude of the phantom.
        '''

        amplitude = self.amplitude

        for i, center in enumerate(self._centers):
            mask = (self._x-center[0])**2+(self._y-center[1])**2 < self._radii[i]**2
            amplitude[mask] = self._amplitudes[i]

        return amplitude
    
    def _prep_phase(self):

        '''
        Prepares the phase of the phantom.
        '''

        phase = self.phase

        for i, center in enumerate(self._centers):
            mask = (self._x-center[0])**2+(self._y-center[1])**2 < self._radii[i]**2
            phase[mask] = self._phases[i]

        return phase
    

class Rectangle(_Phantom):

    def __init__(self, pixelsize: float, shape: Union[List[int], Tuple[int]], 
                 lx: Union[List[float], Tuple[float]], ly: Union[List[float], Tuple[float]], 
                 centers: Union[List[float], Tuple[float]], amplitudes: Union[List[float], Tuple[float]], 
                 phases: Union[List[float], Tuple[float]], background=1.0):

        '''
        Rectangle phantom.

        Parameters
        ----------
        pixelsize: float
            Pixelsize of the phantom.
        shape: List or Tuple
            Shape of the returned phantom field.
        lx: List or Tuple
            List of x-dimensions of the rectangles.
        ly: List or Tuple
            List of y-dimensions of the rectangles.
        centers: List or Tuple
            List of centers of the rectangles.
        amplitudes: List or Tuple
            List of amplitudes of the rectangles.
        phases: List or Tuple
            List of phases of the rectangles.
        background: float
            Background amplitude of the phantom. Default is 1.0.
        '''

        super(Rectangle, self).__init__(
            pixelsize, shape, background, lx, ly, centers, amplitudes, phases
        )

        self._lx = lx
        self._ly = ly
        self._centers = centers
        self._amplitudes = amplitudes
        self._phases = phases
        self._pixelsize = pixelsize
        self._shape = shape

        self.amplitude = self._prep_amplitude()
        self.phase = self._prep_phase()

    def _prep_amplitude(self):

        '''
        Prepares the amplitude of the phantom.
        '''

        amplitude = self.amplitude

        for i, center in enumerate(self._centers):
            mask = (np.abs(self._x-center[0]) < self._lx[i]/2) & (np.abs(self._y-center[1]) < self._ly[i]/2)
            amplitude[mask] = self._amplitudes[i]

        return amplitude
    
    def _prep_phase(self):

        '''
        Prepares the phase of the phantom.
        '''

        phase = self.phase

        for i, center in enumerate(self._centers):
            mask = (np.abs(self._x-center[0]) < self._lx[i]/2) & (np.abs(self._y-center[1]) < self._ly[i]/2)
            phase[mask] = self._phases[i]

        return phase
    
if __name__ == '__main__':

    import matplotlib.pyplot as pp

    params = {
        'wavelength': 550e-9, # wavelength of the light [m]
        'pixelsize': 0.5e-6, # pixel size of the camera [m]
        'magnification': 1, # magnification of the microscope
        'z': 200e-6, # distance from the camera to the object plane [m]
        'z0': -150e-6, # distance from the point source to the object plane [m]
    }

    params_phantom1 = {
        'pixelsize': params['pixelsize']/params['magnification'], # pixel size of the object [m]
        'shape': (512, 512), # shape of the object
        'radii': [5e-6, 7.5e-6, 5e-6], # radius of the circles [m]
        'centers': [
            [-20e-6,-20e-6],
            [  0e-6,  0e-6],
            [ 20e-6, 20e-6]
        ], # centers of the circles [m]
        'amplitudes': [0.8, 0.7, 0.6], # amplitudes of the circles
        'phases': [np.pi/8, np.pi/4, np.pi/2], # phases of the circles
    }

    params_phantom2 = {
        'pixelsize': params['pixelsize']/params['magnification'], # pixel size of the object [m]
        'shape': (512, 512), # shape of the object
        'radii': [5e-6, 5e-6], # radius of the circles [m]
        'centers': [
            [20e-6,-20e-6],
            [-20e-6, 20e-6]
        ], # centers of the circles [m]
        'amplitudes': [0.8, 0.6], # amplitudes of the circles
        'phases': [np.pi/8, np.pi/2], # phases of the circles
    }

    circles1 = Circle(**params_phantom1)
    circles2 = Circle(**params_phantom2)

    # plot
    fig, ax = pp.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(circles1.amplitude, cmap='gray')
    ax[0].set_title('Amplitude')
    ax[1].imshow(circles1.phase, cmap='gray')
    ax[1].set_title('Phase')

    fig, ax = pp.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(circles2.amplitude, cmap='gray')
    ax[0].set_title('Amplitude')
    ax[1].imshow(circles2.phase, cmap='gray')
    ax[1].set_title('Phase')

    circles3 = circles1 * circles2

    fig, ax = pp.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(circles3.amplitude, cmap='gray')
    ax[0].set_title('Amplitude')
    ax[1].imshow(circles3.phase, cmap='gray')
    ax[1].set_title('Phase')
    pp.show()