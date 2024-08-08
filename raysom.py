from typing import List, Sequence, ClassVar

import numpy as np
# import cupy as cp
# import mpmath as mp
import pyopencl as cl
import clinfo, clrenderer

# class PropagatorCp:

#     def __init__(self, amplitude: np.ndarray, phase: np.ndarray, 
#                  pixelsize: Sequence[float],  wavelength: float, 
#                  dtype: np.dtype = np.complex64):
        
#         '''
#         Constructs a scalar electric field propagator that utilizes
#         direct Rayleight-Sommerfeld integration for light difraction.

#         Parameters
#         ----------
#         amplitude: np.ndarray
#             Amplitude or transmittance in the object plane.
#         phase: np.ndarray
#             Phase delay in the object exp(1j*phase)
#         pixel_size: Sequence[float, float]
#             Size of a single pixel in the object plane x times y (m).
#         wavelength: float
#             Wavelength of light (m).
#         dtype: np.dtype
#             Type of the complex data array (hologram). Must be
#             complex64 for single-precision or
#             complex128 for double-precision.
#         '''

#         self._dtype = np.dtype(dtype)
#         self._fp_dtype = {
#             np.complex64: np.dtype(np.float32),
#             np.complex128: np.dtype(np.float64)}.get(self._dtype.type)
#         if self._fp_dtype is None:
#             raise TypeError('Expected dtype np.complex64 or np.complex128 '
#                             'but got {}!'.format(dtype))
        
#         self._pixelsize = cp.array(pixelsize, dtype=self._fp_dtype)
#         self._2pi = cp.multiply(2.0, cp.pi, dtype=self._fp_dtype)
#         self._inv2pi = cp.divide(1.0, self._2pi, dtype=self._fp_dtype)
#         self._k = cp.divide(self._2pi, wavelength, dtype=self._fp_dtype)

#         amplitude = cp.array(amplitude, dtype=self._fp_dtype)
#         phase = cp.array(phase, dtype=self._fp_dtype)

#         self._field = cp.array(amplitude * np.exp(1j*phase), dtype=self._dtype)

#         self._n_eta = amplitude.shape[1]
#         self._n_nu = amplitude.shape[0]

#         eta = cp.arange(-(self._n_eta-1)/2, (self._n_eta-1)/2+1, dtype=self._fp_dtype)
#         eta = cp.multiply(pixelsize[0], eta, dtype=self._fp_dtype)
#         nu = cp.arange(-(self._n_nu-1)/2, (self._n_nu-1)/2+1, dtype=self._fp_dtype)
#         nu = cp.multiply(pixelsize[1], nu, dtype=self._fp_dtype)
#         self._eta, self._nu = cp.meshgrid(eta, nu)

#     def _g_impulse(self, x, y, z):

#         r = cp.sqrt(
#             cp.power(x, 2, dtype=self._fp_dtype) + \
#             cp.power(y, 2, dtype=self._fp_dtype) + \
#             cp.power(z, 2, dtype=self._fp_dtype)
#         )
#         jkr = cp.multiply(1j, self._k*r, dtype=self._dtype)
#         g = self._inv2pi * cp.exp(jkr)/cp.power(r, 3, dtype=self._fp_dtype) * z * cp.subtract(1, jkr, dtype=self._dtype)

#         return g

#     def propagate(self, x, y, z):

#         if x.size != y.size:
#             raise ValueError('x and y must have the same size!')
        
#         if len(x.shape) > 1 or len(y.shape) > 1:
#             raise ValueError('x and y must be 1D arrays!')
        
#         x = cp.array(x, dtype=self._fp_dtype)
#         y = cp.array(y, dtype=self._fp_dtype)
#         z = self._fp_dtype.type(z)
#         _x, _y = cp.meshgrid(x, y)

#         size = _x.size
#         shape = _x.shape
#         field = cp.zeros((size,), dtype=self._dtype)

#         print('Calculating field for {} points...'.format(size))
#         for i, (xi, yi) in enumerate(zip(_x.flat, _y.flat)):
#             g = self._g_impulse(xi - self._eta, yi - self._nu, z)
#             field[i] = cp.sum(self._field * g)
#             print('Progress: {:6.2f}%'.format(100*(i+1)/size), end='\r')

#         field *= self._pixelsize.prod()

#         return field.reshape(shape).get()

class PropagatorCl:

    OPENCL_CODE : ClassVar[str] = '\n'.join([
        '{% if T.float == "float" %}',
        '#define FP(value)   value##f',
        '{%- else -%}',
        '#define FP(value)   value',
        '{%- endif %}',
        '',
        '#define PI          FP(3.141592653589793)',
        '',
        'inline {{ T.float }}2 green_impulse({{ T.float }} x, {{ T.float }} y, {{ T.float }} z, {{ T.float }} w)',
        '{',
        '    {{ T.float }}2 g;',
        '    {{ T.float }} c, s;',
        '',
        '    {{ T.float }} r = sqrt(x*x + y*y + z*z);',
        '    {{ T.float }} k = 2 * PI / w;',
        '    s = sincos(k*r, &c);',
        '',
        '    g.x = 1/(2*PI*r) * z/r * (c/r + s*k);',
        '    g.y = 1/(2*PI*r) * z/r * (s/r - c*k);',
        '',
        '    return g;',
        '}',
        '',
        '__kernel void propagate(',
        '    __global const {{ T.float }} *wavelength, __global const {{ T.float }} *z,',
        '    __global const int *n, __global const {{ T.float }} *deltas, ',
        '    __global const {{ T.float }} *offsets,',
        '    __global const {{ T.float }}2 *field_0,',
        '    __global {{ T.float }} *field_z_real, __global {{ T.float }} *field_z_imag,',
        '    __local {{ T.float }} *ps_real, __local {{ T.float }} *ps_imag)',
        '{',
        '{{ T.float }}2 g;',
        '{{ T.float }} x,y;',
        'int i, j, k, l;',
        'int start, start_old;',
        '',
        'int gid = get_group_id(0);',
        'int lid = get_local_id(0);',
        'int group_size = get_local_size(0);',
        'int in_field_size = n[2] * n[3];',
        'int remaining_size = in_field_size;',
        '',
        'j = gid % n[0];',
        'i = gid / n[0];',
        '',
        'for (uint tlid = lid; tlid < in_field_size; tlid += group_size)',
        '    {',
        '        remaining_size -= group_size;',
        '        if (remaining_size < 0)',
        '        {',
        '            group_size = group_size + remaining_size;',
        '        }',
        '',
        '        l = tlid % n[2];',
        '        k = tlid / n[2];',
        '',
        '        x = offsets[0] + j*deltas[0];',
        '        x = x - (offsets[2] + l*deltas[2]);',
        '        y = offsets[1] + i*deltas[1];',
        '        y = y - (offsets[3] + k*deltas[3]);',
        '',
        '        g = green_impulse(x, y, *z, *wavelength);',
        '',
        '        ps_real[lid] = deltas[2] * deltas[3] *',
        '            (field_0[tlid].x * g.x - field_0[tlid].y * g.y);',
        '        ps_imag[lid] = deltas[2] * deltas[3] *',
        '            (field_0[tlid].x * g.y + field_0[tlid].y * g.x);',
        '',
        '        start = group_size % 2;',
        '        for (uint stride = group_size/2; stride > 0; stride = (start_old + stride)/2)',
        '        {',
        '            barrier(CLK_LOCAL_MEM_FENCE);',
        '            if (lid >= start && lid < stride + start)',
        '            {',
        '                ps_real[lid] += ps_real[lid + stride];',
        '                ps_imag[lid] += ps_imag[lid + stride];',
        '            }',
        '            start_old = start;',
        '            start = (start + stride) % 2;',
        '        }',
        '        barrier(CLK_GLOBAL_MEM_FENCE);',
        '        if (lid == 0)',
        '        {',
        '            field_z_real[gid] += ps_real[0];',
        '            field_z_imag[gid] += ps_imag[0];',
        '        }',
        '    }',
        '}',
    ])

    def __init__(self, amplitude: np.ndarray, phase: np.ndarray, 
                 pixelsize: Sequence[float],  wavelength: float, 
                 dtype: np.dtype = np.complex64):
        
        '''
        Constructs a scalar electric field propagator that utilizes
        direct Rayleight-Sommerfeld integration for light difraction.

        Parameters
        ----------
        amplitude: np.ndarray
            Amplitude or transmittance in the object plane.
        phase: np.ndarray
            Phase delay in the object exp(1j*phase)
        pixel_size: Sequence[float, float]
            Size of a single pixel in the object plane x times y (m).
        wavelength: float
            Wavelength of light (m).
        dtype: np.dtype
            Type of the complex data array (hologram). Must be
            complex64 for single-precision or
            complex128 for double-precision.
        '''

        self._dtype = np.dtype(dtype)
        self._fp_dtype = {
            np.complex64: np.dtype(np.float32),
            np.complex128: np.dtype(np.float64)}.get(self._dtype.type)
        if self._fp_dtype is None:
            raise TypeError('Expected dtype np.complex64 or np.complex128 '
                            'but got {}!'.format(dtype))

        self._wavelength = float(wavelength)
        self._size = int(np.prod(amplitude.shape))
        self._pixelsize = np.array(pixelsize, dtype=self._fp_dtype)

        self._field = np.array(amplitude * np.exp(1j*phase), dtype=self._dtype)
        self._amplitude = np.array(amplitude, dtype=self._fp_dtype)
        self._phase = np.array(phase, dtype=self._fp_dtype)

        self._n_eta = amplitude.shape[1]
        self._n_nu = amplitude.shape[0]

        self._eta = pixelsize[0] * np.arange(-(self._n_eta-1)/2, (self._n_eta-1)/2+1)
        self._nu = pixelsize[1] * np.arange(-(self._n_nu-1)/2, (self._n_nu-1)/2+1)

        self._gpu = False

    def init_gpu(self, cldevices: List[cl.Device] | cl.Context | cl.CommandQueue = None,
        clbuild_options: List[str] = None):

        self._gpu = True
        self._ctx = self._cmdqueue = None

        # select the first GPU device by default
        if cldevices is None:
            cldevices = [clinfo.gpus()[0]]

        # check if we were given an OpenCL context instead of devices
        if isinstance(cldevices, cl.Context):
            self._ctx = cldevices
        
        # check if we were given an OpenCL command queue instead of devices
        if isinstance(cldevices, cl.CommandQueue):
            self._cmdqueue = cldevices
            self._ctx = self._cmdqueue.context

        # creating OpenCL context and command queue if required
        if self._ctx is None:
            self._ctx = cl.Context(cldevices)
        if self._cmdqueue is None:
            self._cmdqueue= cl.CommandQueue(self._ctx)

        cldevice = self._cmdqueue.get_info(cl.command_queue_info.DEVICE)
        self._MAX_WG_SIZE = cldevice.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

        # creating and rendering the template from the source code
        self._program = clrenderer.render(self._ctx, clbuild_options,
            self.OPENCL_CODE, self._dtype)

        # creating related OpenCL buffers
        self._np_wavelength = np.array([self._wavelength], dtype=self._fp_dtype)
        self._np_z = np.array([0], dtype=self._fp_dtype)
        self._np_n = np.array([0, 0, 0, 0], dtype=np.int32)
        self._np_deltas = np.array([0, 0, 0, 0], dtype=self._fp_dtype)
        self._np_offsets = np.array([0, 0, 0, 0], dtype=self._fp_dtype)
        self._np_field_0 = np.empty((self._n_nu, self._n_eta, 2), dtype=self._fp_dtype)
        self._np_field_0[:,:,0] = self._field.real
        self._np_field_0[:,:,1] = self._field.imag

        r = cl.mem_flags.READ_ONLY
        chp = cl.mem_flags.COPY_HOST_PTR
        self._cl_wavelength = cl.Buffer(self._ctx, r | chp, hostbuf=self._np_wavelength)
        self._cl_z = cl.Buffer(self._ctx, r | chp, hostbuf=self._np_z)
        self._cl_n = cl.Buffer(self._ctx, r | chp, hostbuf=self._np_n)
        self._cl_deltas = cl.Buffer(self._ctx, r | chp, hostbuf=self._np_deltas)
        self._cl_offsets = cl.Buffer(self._ctx, r | chp, hostbuf=self._np_offsets)
        self._cl_np_field_0 = cl.Buffer(self._ctx, r | chp, hostbuf=self._np_field_0)

        cl.enqueue_copy(self._cmdqueue, self._cl_wavelength, self._np_wavelength)
        cl.enqueue_copy(self._cmdqueue, self._cl_np_field_0, self._np_field_0)

    def propagate(self, x: np.ndarray, y: np.ndarray, z: float):

        if x.size != y.size:
            raise ValueError('x and y must have the same size!')
        
        if len(x.shape) > 1 or len(y.shape) > 1:
            raise ValueError('x and y must be 1D arrays!')
      
        pixelsize = [0,0]
        try:
            pixelsize[0] = x[1] - x[0]
        except IndexError:
            pass
        try:
            pixelsize[1] = y[1] - y[0]
        except IndexError:
            pass

        if self._gpu:
            self._np_z[0] = z
            self._np_n = np.array([x.size, y.size, self._n_eta, self._n_nu], dtype=np.int32)
            self._np_deltas = np.array([pixelsize[0],pixelsize[1],self._pixelsize[0], self._pixelsize[1]], dtype=self._fp_dtype)
            self._np_offsets = np.array([x[0], y[0], self._eta[0], self._nu[0]], dtype=self._fp_dtype)

            cl.enqueue_copy(self._cmdqueue, self._cl_z, self._np_z)
            cl.enqueue_copy(self._cmdqueue, self._cl_n, self._np_n)
            cl.enqueue_copy(self._cmdqueue, self._cl_deltas, self._np_deltas)
            cl.enqueue_copy(self._cmdqueue, self._cl_offsets, self._np_offsets)

            rw = cl.mem_flags.READ_WRITE
            chp = cl.mem_flags.COPY_HOST_PTR
            np_field_z_real = np.zeros([y.size,x.size], dtype=self._fp_dtype)
            np_field_z_imag = np.zeros([y.size,x.size], dtype=self._fp_dtype)
            cl_field_z_real = cl.Buffer(self._ctx, rw | chp, hostbuf=np_field_z_real)
            cl_field_z_imag = cl.Buffer(self._ctx, rw | chp, hostbuf=np_field_z_imag)
            local_ps_real = cl.LocalMemory(np.empty((self._MAX_WG_SIZE,),dtype=self._fp_dtype).nbytes)
            local_ps_imag = cl.LocalMemory(np.empty((self._MAX_WG_SIZE,),dtype=self._fp_dtype).nbytes)

            work_items = x.size * y.size * min(self._MAX_WG_SIZE, self._n_eta * self._n_nu)
            wg_size = min(self._MAX_WG_SIZE, self._n_eta * self._n_nu)

            self._program.propagate(self._cmdqueue, (work_items,), (wg_size,),
                self._cl_wavelength, self._cl_z, self._cl_n, self._cl_deltas, self._cl_offsets,
                self._cl_np_field_0, cl_field_z_real, cl_field_z_imag, local_ps_real, local_ps_imag)
            
            cl.enqueue_copy(self._cmdqueue, np_field_z_real, cl_field_z_real)
            cl.enqueue_copy(self._cmdqueue, np_field_z_imag, cl_field_z_imag)

            return np_field_z_real + np.multiply(1j, np_field_z_imag, dtype=self._dtype)
        else:
            return np.empty((y.size,), dtype=self._dtype)
        
class PropagatorNp:

    def __init__(self, amplitude: np.ndarray, phase: np.ndarray, 
                 pixelsize: Sequence[float],  wavelength: float, 
                 dtype: np.dtype = np.complex64):
        
        '''
        Constructs a scalar electric field propagator that utilizes
        direct Rayleight-Sommerfeld integration for light difraction.

        Parameters
        ----------
        amplitude: np.ndarray
            Amplitude or transmittance in the object plane.
        phase: np.ndarray
            Phase delay in the object exp(1j*phase)
        pixel_size: Sequence[float, float]
            Size of a single pixel in the object plane x times y (m).
        wavelength: float
            Wavelength of light (m).
        dtype: np.dtype
            Type of the complex data array (hologram). Must be
            complex64 for single-precision or
            complex128 for double-precision.
        '''

        self._dtype = np.dtype(dtype)

        self._fp_dtype = {
            np.complex64: np.dtype(np.float32),
            np.complex128: np.dtype(np.float64)}.get(self._dtype.type)
        
        if self._fp_dtype is None:
            raise TypeError('Expected dtype np.complex64 or np.complex128 '
                            'but got {}!'.format(dtype))

        self._2pi = np.multiply(2.0, np.pi, dtype=self._fp_dtype)
        self._inv2pi = np.divide(1.0, self._2pi, dtype=self._fp_dtype)
        self._k = np.divide(self._2pi, wavelength, dtype=self._fp_dtype)
        self._pixelsize = np.array(pixelsize, dtype=self._fp_dtype)

        self._field = np.array(amplitude * np.exp(1j*phase), dtype=self._dtype)
        self._amplitude = np.array(amplitude, dtype=self._fp_dtype)
        self._phase = np.array(phase, dtype=self._fp_dtype)

        self._n_eta = amplitude.shape[1]
        self._n_nu = amplitude.shape[0]

        self._eta = np.arange(
            -(self._n_eta-1)/2, (self._n_eta-1)/2+1, dtype=self._fp_dtype
        )
        self._eta = np.multiply(self._pixelsize[0], self._eta, dtype=self._fp_dtype)

        self._nu = np.arange(
            -(self._n_nu-1)/2, (self._n_nu-1)/2+1, dtype=self._fp_dtype
        )
        self._nu = np.multiply(self._pixelsize[1], self._nu, dtype=self._fp_dtype)

    def _green_impulse(self, x, y, z):

        r = np.sqrt(
            np.power(x, 2, dtype=self._fp_dtype) + \
            np.power(y, 2, dtype=self._fp_dtype) + \
            np.power(z, 2, dtype=self._fp_dtype)
        )
        jkr = np.multiply(1j, self._k*r, dtype=self._dtype)
        g = self._inv2pi * np.exp(jkr)/np.power(r, 3, dtype=self._fp_dtype) * z * np.subtract(1, jkr, dtype=self._dtype)

        return g

    def propagate(self, x: np.ndarray, y: np.ndarray, z: float):

        if x.size != y.size:
            raise ValueError('x and y must have the same size!')
        
        if len(x.shape) > 1 or len(y.shape) > 1:
            raise ValueError('x and y must be 1D arrays!')

        x = np.array(x, dtype=self._fp_dtype)
        y = np.array(y, dtype=self._fp_dtype)
        z = self._fp_dtype.type(z)

        g = self._green_impulse(
            x[np.newaxis,:,np.newaxis,np.newaxis] - self._eta[np.newaxis,np.newaxis,np.newaxis,:],
            y[:,np.newaxis,np.newaxis,np.newaxis] - self._nu[np.newaxis,np.newaxis,:,np.newaxis],
            z
        )

        field = np.sum(self._field[np.newaxis,np.newaxis,:,:] * g, axis=(2,3))
        field *= self._pixelsize.prod()

        return field
    
# class PropagatorMp:

#     def __init__(self, amplitude: np.ndarray, phase: np.ndarray, 
#                  pixelsize: Sequence[float],  wavelength: float, 
#                  dp: 10):
        
#         '''
#         Constructs a scalar electric field propagator that utilizes
#         direct Rayleight-Sommerfeld integration for light difraction.

#         Parameters
#         ----------
#         amplitude: np.ndarray
#             Amplitude or transmittance in the object plane.
#         phase: np.ndarray
#             Phase delay of the object exp(1j*phase)
#         pixel_size: Sequence[float, float]
#             Size of a single pixel in the object plane width by height (m).
#         wavelength: float
#             Wavelength of light (m).
#         dp: int
#             Decimal places for the mpmath library.
#             Default is 10.
#         '''
#         mp.dps = dp

#         self._wavelength = mp.mpf(wavelength)
#         self._pixelsize = mp.matrix(pixelsize)

#         field = amplitude * np.exp(1j*phase)
#         self._field = mp.matrix(field.tolist())

#         self._n_eta = int(amplitude.shape[1])
#         self._n_nu = int(amplitude.shape[0])

#         eta = pixelsize[0] * np.arange(-(self._n_eta-1)/2, (self._n_eta-1)/2+1)
#         self._eta = mp.matrix(eta.tolist())
#         nu = pixelsize[1] * np.arange(-(self._n_nu-1)/2, (self._n_nu-1)/2+1)
#         self._nu = mp.matrix(nu.tolist())

#     def _green_impulse(self, x, y, z):
#         k = 2 * mp.pi / self._wavelength
#         r = mp.sqrt(x**2 + y**2 + z**2)
#         g = 1/(2*mp.pi) * mp.expj(k*r)/r * z/r**2 * (1 - 1j*k*r)

#         return g
    
#     def propagate(self, x: np.ndarray, y: np.ndarray, z: float):

#         if x.size != y.size:
#             raise ValueError('x and y must have the same size!')
        
#         if len(x.shape) > 1 or len(y.shape) > 1:
#             raise ValueError('x and y must be 1D arrays!')

#         field = mp.zeros(y.size, x.size)

#         print('Calculating field for {} points...'.format(x.size*y.size))

#         for i, xi in enumerate(x):
#             for j, yj in enumerate(y):
#                 print('Progress: {:6.2f}%'.format(100*(i*y.size+j)/(x.size*y.size)), end='\r')
#                 for k, eta in enumerate(self._eta):
#                     for l, nu in enumerate(self._nu):
#                         g = self._green_impulse(
#                             xi - eta,
#                             yj - nu,
#                             z
#                         )
#                         field[j,i] += self._field[l,k] * g

#         field *= self._pixelsize[0] * self._pixelsize[1]
#         return field