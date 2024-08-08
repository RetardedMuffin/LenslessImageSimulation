import clinfo
import scalarwave as sw

from typing import List, Tuple, Sequence, ClassVar
import copy

import numpy as np
from matplotlib import pyplot as pp
import reikna.cluda as cluda
import reikna.fft as clfft
import pyopencl as cl
import jinja2

class Propagator:
    OPENCL_CODE : ClassVar[str] = '\n'.join([
        '{% if T.float == "float" %}',
        '#define FP(value)   value##f',
        '{%- else -%}',
        '#define FP(value)   value',
        '{%- endif %}',
        '',
        '#define PI          FP(3.141592653589793)',
        '#define FP_0        FP(0.0)',
        '#define FP_0p5      FP(0.5)',
        '#define FP_1        FP(1.0)',
        '#define FP_2        FP(2.0)',
        '#define COND_1_NA  FP({{T.cond_1_na}})',
        '',
        '__kernel void field_propagate(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out,',
        '    __global {{ T.float }} const *fx,',
        '    __global {{ T.float }} const *fy,',
        '    __global {{ T.float }} const *z,',
        '    __global {{ T.float }} const *wavelength)',
        '{',
        '    size_t i = get_global_id(0);',
        '    {{ T.float }} h_x, h_y;',
        '    {{ T.float }}2 h_times_field;',
        '',
        '    {{ T.float }} arg = FP_1 - (*wavelength)*(*wavelength)*(fx[i]*fx[i] + ',
        '       fy[i]*fy[i]);',
        '    if (arg < COND_1_NA)',
        '    {',
        '        out[i].x = FP_0;',
        '        out[i].y = FP_0;',
        '    }',
        '    else',
        '    {',
        '        arg = (FP_2 * PI) * (*z) / (*wavelength) *',
        '            sqrt(arg);',
        '        h_y = sincos(arg, &h_x);',
        '        h_times_field.x = field[i].x*h_x - field[i].y*h_y;',
        '        h_times_field.y = field[i].x*h_y + field[i].y*h_x;',
        '        out[i] = h_times_field;',
        '    }',
        '};',
        '',
        '__kernel void field_backpropagate(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out,',
        '    __global {{ T.float }} const *fx,',
        '    __global {{ T.float }} const *fy,',
        '    __global {{ T.float }} const *z,',
        '    __global {{ T.float }} const *wavelength)',
        '{',
        '    size_t i = get_global_id(0);',
        '    {{ T.float }} h_x, h_y;',
        '    {{ T.float }}2 h_times_field;',
        '',
        '    {{ T.float }} arg = FP_1 - (*wavelength)*(*wavelength)*(fx[i]*fx[i] + ',
        '       fy[i]*fy[i]);',
        '    if (arg < COND_1_NA)',
        '    {',
        '        out[i].x = FP_0;',
        '        out[i].y = FP_0;',
        '    }',
        '    else',
        '    {',
        '        arg = -(FP_2 * PI) * (*z) / (*wavelength) *',
        '            sqrt(arg);',
        '        h_y = sincos(arg, &h_x);',
        '        h_times_field.x = field[i].x*h_x - field[i].y*h_y;',
        '        h_times_field.y = field[i].x*h_y + field[i].y*h_x;',
        '        out[i] = h_times_field;',
        '    }',
        '};',
        '',
        '__kernel void constrain_nonneg(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out)',
        '    {',
        '        size_t i = get_global_id(0);',
        '',
        '        {{ T.float }} amp = sqrt(field[i].x*field[i].x +',
        '            field[i].y*field[i].y);',
        '',
        '        if (amp > FP_1)',
        '        {',
        '            out[i].x = FP_1;',
        '            out[i].y = FP_0;',
        '        }',
        '        else',
        '        {',
        '            out[i] = field[i];',
        '        }',
        '    };',
        '',
        '__kernel void constrain_hologram(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out,',
        '    __global {{ T.float }} const *amplitude)',
        '    {',
        '        size_t i = get_global_id(0);',
        '        {{ T.float }} c, s;',
        '',
        '        {{ T.float }} phi = atan2(field[i].y, field[i].x);',
        '        s = sincos(phi, &c);',
        '',
        '        out[i].x = amplitude[i] * c;',
        '        out[i].y = amplitude[i] * s;',
        '    };',
    ])

    def __init__(self, amplitude: np.ndarray, phase: np.ndarray, pixelsize: float,
        wavelength: float, dtype: np.dtype = np.complex64, default_fft_shift: bool = True,
        NA: float = 1.0, n_med: float = 1.0):

        '''
        Constructs a scalar electric field propagator that utilizes
        angular spectrum for propagation of light.

        Parameters
        ----------
        amplitude: np.ndarray
            Real transmittance of the object.
        phase: np.ndarray
            Phase delay of the object exp(1j*phase)
        pixelsize: float
            Pixel size of the propagated object (m).
        wavelength: float
            Wavelength of light (m).
        dtype: np.dtype
            Type of the complex data array (hologram). Must be
            complex64 for single-precision or
            complex128 for double-precision.
        default_fft_shift: bool
            If True, the spatial frequencies are shifted so that the
            FFT of field/hologram does not have to be shifted. If False,
            the hologram is shifted after applying FFT and before applying
            inverse FFT.
        NA: float
            Numerical aperture of the imaging system.
        n_med: float
            Refractive index of the medium.
        '''

        self._pixelsize = float(pixelsize)
        self._wavelength = float(wavelength)
        self._size = int(np.prod(amplitude.shape))
        self._cond_1_na = 1 - (NA/n_med)**2
        self._dtype = np.dtype(dtype)
        self._fp_dtype = {
            np.complex64: np.dtype(np.float32),
            np.complex128: np.dtype(np.float64)}.get(self._dtype.type)
        if self._fp_dtype is None:
            raise TypeError('Expected dtype np.complex64 or np.complex128 '
                            'but got {}!'.format(dtype))

        # arrays for complex data field
        self._orig_amplitude = amplitude
        self._orig_phase = phase

        self._field = np.array(amplitude * np.exp(1j*phase), dtype=self._dtype)
        self._amplitude = np.array(amplitude, dtype=self._fp_dtype)

        # arrays of spatial frequencies (shifted to match FFT output)
        fx = np.array(np.fft.fftfreq(self._field.shape[1], self._pixelsize), dtype=self._fp_dtype)
        fy = np.array(np.fft.fftfreq(self._field.shape[0], self._pixelsize), dtype=self._fp_dtype)
        if default_fft_shift:
            self._fx, self._fy = np.meshgrid(
                fx, fy
            )
        else:
            self._fx, self._fy = np.meshgrid(
                np.fft.fftshift(fx), np.fft.fftshift(fy)
            )

        # by default the usage of devices (gpus) is turned off
        self._gpu = False
        self._default_fft_shift = default_fft_shift
        
        self._xcoors, self._ycoors = self._coordinates()
        self._backpropagate_sequential = False
        self._F_field = None

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

        # preparing reikna/cluda OpenCL kernels
        self._cluda_api = cluda.ocl_api()
        self._reikna_thread = self._cluda_api.Thread(self._cmdqueue)
        # creating kernel for computing FFT and inverse FFT
        self._cluda_field = self._reikna_thread.empty_like(self._field)
        self._cluda_field_fft = self._reikna_thread.empty_like(self._field)
        self._reikna_fft = clfft.FFT(self._field).compile(
            self._reikna_thread, compiler_options=clbuild_options)
        # creating kernel for shifting the FFT of complex field
        self._reikna_fft_shift = clfft.FFTShift(self._field).compile(
            self._reikna_thread, compiler_options=clbuild_options)

        # rendering OpenCL code from the jinja template
        # creating context data - just a single parameter T.float
        render_ctx = {
            np.complex64: {'T':{'float': 'float', 'cond_1_na': self._cond_1_na}},
            np.complex128: {'T':{'float': 'double', 'cond_1_na': self._cond_1_na}}
        }.get(self._dtype.type)
        # creating and rendering the template from the source code
        self._cl_source = jinja2.Template(
            self.OPENCL_CODE).render(**render_ctx)
        # building/compiling the OpenCL source code
        self._program = cl.Program(self._ctx, self._cl_source).build(
            options=clbuild_options)

        mf = cl.mem_flags

        # creating related OpenCL buffers
        self._cl_fx = cl.Buffer(self._ctx,
                                mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=self._fx)
        self._cl_fy = cl.Buffer(self._ctx,
                                mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=self._fy)

        # hologram intensity - amplitude
        self._cl_amplitude = cl.Buffer(self._ctx,
                                mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=self._amplitude)

        # scalar arguments of the propagation kernel
        self._np_wavelength = np.array((self._wavelength,), dtype=self._fp_dtype)
        self._np_z = np.zeros((1,), dtype=self._fp_dtype)
        self._cl_wavelength = cl.Buffer(self._ctx,
                                        mf.READ_WRITE | mf.COPY_HOST_PTR,
                                        hostbuf=self._np_wavelength)
        self._cl_z = cl.Buffer(self._ctx,
                               mf.READ_WRITE | mf.COPY_HOST_PTR,
                               hostbuf=self._np_z)

        # preparing the field propagation callable with premapped arguments
        self._field_propagate_call = self._program.field_propagate
        self._field_propagate_call.set_args(
            self._cluda_field_fft.base_data,
            self._cluda_field_fft.base_data,
            self._cl_fx,
            self._cl_fy,
            self._cl_z,
            self._cl_wavelength)

        # preparing the field backpropagation callable with premapped arguments
        self._field_backpropagate_call = self._program.field_backpropagate
        self._field_backpropagate_call.set_args(
            self._cluda_field_fft.base_data,
            self._cluda_field_fft.base_data,
            self._cl_fx,
            self._cl_fy,
            self._cl_z,
            self._cl_wavelength)

        # preparing the absorption nonnegativity constrain for object
        self._constrain_nonneg_call = self._program.constrain_nonneg
        self._constrain_nonneg_call.set_args(
            self._cluda_field.base_data,
            self._cluda_field.base_data)

        # preparing the hologram constrain for image plane
        self._constrain_hologram_call = self._program.constrain_hologram
        self._constrain_hologram_call.set_args(
            self._cluda_field.base_data,
            self._cluda_field.base_data,
            self._cl_amplitude)

        # host to device transfer of amplitude, field, frequencies, z and wavelength
        cl.enqueue_copy(self._cmdqueue, self._cl_amplitude, self._amplitude)
        cl.enqueue_copy(self._cmdqueue, self._cluda_field.base_data, self._field)
        cl.enqueue_copy(self._cmdqueue, self._cl_fx, self._fx)
        cl.enqueue_copy(self._cmdqueue, self._cl_fy, self._fy)
        cl.enqueue_copy(self._cmdqueue, self._cl_z, self._np_z)
        cl.enqueue_copy(self._cmdqueue, self._cl_wavelength, self._np_wavelength)

    def reset(self):
        self.field = np.array(self._orig_amplitude * np.exp(1j*self._orig_phase), dtype=self._dtype)

    @property
    def wavelength(self) -> float:

        '''
        Wavelength (in m)
        '''

        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value: float):

        self._wavelength = float(value)

        if self._gpu:
            self._np_wavelength = np.array((self._wavelength,), dtype=self._fp_dtype)
            cl.enqueue_copy(self._cmdqueue, self._cl_wavelength, self._np_wavelength)

    @property
    def field(self) -> np.ndarray:

        '''
        Scalar electric field.
        '''

        if self._gpu:
            cl.enqueue_copy(self._cmdqueue, self._field, self._cluda_field.base_data)

        return self._field

    @field.setter
    def field(self, value: np.ndarray):

        if value.shape != self._field.shape:
            raise ValueError(
                'Input field is not the same ' + \
                'shape as allocated field'
            )
        else:
            self._field = value

        if self._gpu:
            cl.enqueue_copy(self._cmdqueue, self._cluda_field.base_data, self._field)

    @property
    def amplitude(self) -> np.ndarray:

        '''
        Amplitude of the field.
        '''

        return np.abs(self.field)

    @property
    def phase(self) -> np.ndarray:

        '''
        Phase of the field within the interval [-pi, pi]
        (unwrapping required)
        '''

        return np.angle(self.field)

    @property
    def X(self):

        '''
        Mesh of x coordinates.
        '''

        return self._xcoors

    @property
    def Y(self):

        '''
        Mesh of y coordinates.
        '''

        return self._ycoors

    def _coordinates(self) -> Tuple[np.ndarray, np.ndarray]:

        '''
        Pixel coordinate grids for the object and propagated fields.

        Returns
        -------
        out: tuple
            Tuple of x any y coordinate grids.
        '''

        xcoors, ycoors = np.meshgrid(
            self._pixelsize * \
                np.arange(-np.floor(self._field.shape[1]/2), 
                    np.ceil(self._field.shape[1]/2)),
            self._pixelsize * \
                np.arange(-np.floor(self._field.shape[0]/2), 
                    np.ceil(self._field.shape[0]/2)))

        return xcoors, ycoors

    def backpropagate(self, z: float):

        '''
        Backpropagates the field for a distance z.

        Parameters
        ----------
        z: float
            Distance by which field is backpropagated (m).

        '''

        if self._gpu:
            if self._np_z[0] != z:
                self._np_z[0] = z
                cl.enqueue_copy(self._cmdqueue,
                    self._cl_z, self._np_z)
            self._fft2gpu()
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._field_backpropagate_call,
                [self._size], None)
            self._ifft2gpu()
        else:
            F_field = self._fft2(self._field)
            H = self._transfer_function(-z)
            F_field_H = F_field * H
            self._field = self._ifft2(F_field_H)
    
    def propagate(self, z: float):

        '''
        Propagates the field for a distance z.

        Parameters
        ----------
        z: float
            Distance by which field is propagated (m).

        '''

        if self._gpu:
            if self._np_z[0] != z:
                self._np_z[0] = z
                cl.enqueue_copy(self._cmdqueue,
                    self._cl_z, self._np_z)
            self._fft2gpu()
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._field_propagate_call,
                [self._size], None)
            self._ifft2gpu()
        else:
            F_field = self._fft2(self._field)
            H = self._transfer_function(z)
            F_field_H = F_field * H
            self._field = self._ifft2(F_field_H)

    def backpropagate_sequential(self, dz: np.ndarray):

        '''
        Sequentially backpropagates the field by a distance dz.
        First do a backpropagate

        Parameters
        ----------
        dz: np.ndarray
            Distances by which the field is sequentially
            backpropagated (m).

        '''

        if not self._backpropagate_sequential:
            if self._gpu:
                self._fft2gpu()
            else:
                self._F_field = self._fft2(self._field)

        self._backpropagate_sequential = True

        if self._gpu:
            self._np_z[0] = dz
            cl.enqueue_copy(self._cmdqueue,
                self._cl_z, self._np_z)
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._field_backpropagate_call,
                [self._size], None)
            self._ifft2gpu()
            if not self._default_fft_shift:
                self._reikna_fft_shift(self._cluda_field_fft, self._cluda_field_fft)
        else:
            H = self._transfer_function(-dz)
            self._F_field = self._F_field * H
            self._field = self._ifft2(self._F_field)
            if not self._default_fft_shift:
                self._F_field = np.fft.ifftshift(self._F_field)            

        return self.field

    def _transfer_function(self, z: float) -> np.ndarray:

        '''
        Calculates the transfer function for
        the angular spectrum. For backpropagation 
        use -z.

        Parameters
        ----------
        z: float
            Distance by which field is propagated (m).
        '''

        arg = 1 - self._wavelength**2 * (self._fx**2 + self._fy**2)
        H = np.exp(1j * 2 * np.pi * z / self._wavelength * np.sqrt(arg))
        H[arg < self._cond_1_na] = 0.0

        return H

    def apply_constrain_nonneg(self):

        '''
        Applies the absorption non-negativity
        constrain that can be utilized for
        iterative reconstruction.
        '''

        if self._gpu:
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._constrain_nonneg_call,
                [self._size], None)
        else:

            constrain_mask = self.amplitude > 1.0

            amplitude = np.array(self.amplitude)
            amplitude[constrain_mask] = 1.0

            phase = np.array(self.phase)
            phase[constrain_mask] = 0.0

            self.field = amplitude * np.exp(1j*phase)

    def apply_constrain_hologram(self):
        
        '''
        Applies the hologram constrain in the image
        plane.
        '''

        if self._gpu:
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._constrain_hologram_call,
                [self._size], None)
        else:
            self.field = self._amplitude * np.exp(1j*self.phase)

    def _fft2gpu(self):
        self._reikna_fft(self._cluda_field_fft, self._cluda_field, 0)
        if not self._default_fft_shift:
            self._reikna_fft_shift(self._cluda_field_fft, self._cluda_field_fft)

    def _fft2(self, field):
        if self._default_fft_shift:
            return np.fft.fft2(field)
        else:
            return np.fft.fftshift(np.fft.fft2(field))

    def _ifft2gpu(self):
        if not self._default_fft_shift:
            self._reikna_fft_shift(self._cluda_field_fft, self._cluda_field_fft)
        self._reikna_fft(self._cluda_field, self._cluda_field_fft, 1)

    def _ifft2(self, F):
        if self._default_fft_shift:
            return np.fft.ifft2(F)
        else:
            return np.fft.ifft2(np.fft.ifftshift(F))

class PropagatorSphericalWave:
    OPENCL_CODE : ClassVar[str] = '\n'.join([
        '{% if T.float == "float" %}',
        '#define FP(value)   value##f',
        '{%- else -%}',
        '#define FP(value)   value',
        '{%- endif %}',
        '',
        '#define PI          FP(3.141592653589793)',
        '#define FP_0        FP(0.0)',
        '#define FP_0p5      FP(0.5)',
        '#define FP_1        FP(1.0)',
        '#define FP_2        FP(2.0)',
        '#define COND_1_NA  FP({{T.cond_1_na}})',
        '',
        '__kernel void field_propagate(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out,',
        '    __global {{ T.float }} const *fx,',
        '    __global {{ T.float }} const *fy,',
        '    __global {{ T.float }} const *z,',
        '    __global {{ T.float }} const *wavelength)',
        '{',
        '    size_t i = get_global_id(0);',
        '    {{ T.float }} h_x, h_y;',
        '    {{ T.float }}2 h_times_field;',
        '',
        '    {{ T.float }} arg = FP_1 - (*wavelength)*(*wavelength)*(fx[i]*fx[i] + ',
        '       fy[i]*fy[i]);',
        '    if (arg < COND_1_NA)',
        '    {',
        '        out[i].x = FP_0;',
        '        out[i].y = FP_0;',
        '    }',
        '    else',
        '    {',
        '        arg = (FP_2 * PI) * (*z) / (*wavelength) *',
        '            sqrt(arg);',
        '        h_y = sincos(arg, &h_x);',
        '        h_times_field.x = field[i].x*h_x - field[i].y*h_y;',
        '        h_times_field.y = field[i].x*h_y + field[i].y*h_x;',
        '        out[i] = h_times_field;',
        '    }',
        '};',
        '',
        '__kernel void pre_propagation_sw(',
        '    __global {{ T.float }}2 const *sw_0,',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out)',
        '{',
        '    size_t i = get_global_id(0);',
        '    {{ T.float }}2 field_tmp;',
        '    field_tmp.x = (field[i].x - FP_1) * sw_0[i].x - field[i].y * sw_0[i].y;',
        '    field_tmp.y = (field[i].x - FP_1) * sw_0[i].y + field[i].y * sw_0[i].x;',
        # '    field_tmp.x = sw_0[i].x;',
        # '    field_tmp.y = sw_0[i].y;',
        '    out[i] = field_tmp;',
        '};',
        '',
        '__kernel void post_propagation_sw(',
        '    __global {{ T.float }}2 const *sw_z,',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out)',
        '{',
        '    size_t i = get_global_id(0);',
        '    out[i].x = field[i].x + sw_z[i].x;',
        '    out[i].y = field[i].y + sw_z[i].y;',
        '};',
        '',
        '__kernel void field_backpropagate(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out,',
        '    __global {{ T.float }} const *fx,',
        '    __global {{ T.float }} const *fy,',
        '    __global {{ T.float }} const *z,',
        '    __global {{ T.float }} const *wavelength)',
        '{',
        '    size_t i = get_global_id(0);',
        '    {{ T.float }} h_x, h_y;',
        '    {{ T.float }}2 h_times_field;',
        '',
        '    {{ T.float }} arg = FP_1 - (*wavelength)*(*wavelength)*(fx[i]*fx[i] + ',
        '       fy[i]*fy[i]);',
        '    if (arg < COND_1_NA)',
        '    {',
        '        out[i].x = FP_0;',
        '        out[i].y = FP_0;',
        '    }',
        '    else',
        '    {',
        '        arg = -(FP_2 * PI) * (*z) / (*wavelength) *',
        '            sqrt(arg);',
        '        h_y = sincos(arg, &h_x);',
        '        h_times_field.x = field[i].x*h_x - field[i].y*h_y;',
        '        h_times_field.y = field[i].x*h_y + field[i].y*h_x;',
        '        out[i] = h_times_field;',
        '    }',
        '};',
        '',
        '__kernel void pre_backpropagation_sw(',
        '    __global {{ T.float }}2 const *sw_z,',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out)',
        '{',
        '    size_t i = get_global_id(0);',
        '    out[i].x = field[i].x - sw_z[i].x;',
        '    out[i].y = field[i].y - sw_z[i].y;',
        '};',
        '',
        '__kernel void post_backpropagation_sw(',
        '    __global {{ T.float }}2 const *sw_0,',
        '    __global {{ T.float }} const *div_amp_0,',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out)',
        '{',
        '    size_t i = get_global_id(0);',
        '    {{ T.float }}2 field_tmp;',
        '    field_tmp.x = (field[i].x * sw_0[i].x + field[i].y * sw_0[i].y) * div_amp_0[i] + FP_1;',
        '    field_tmp.y = (field[i].y * sw_0[i].x - field[i].x * sw_0[i].y) * div_amp_0[i];',
        '    out[i] = field_tmp;',
        '};',
        '',
        '__kernel void constrain_nonneg(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out)',
        '    {',
        '        size_t i = get_global_id(0);',
        '',
        '        {{ T.float }} amp = sqrt(field[i].x*field[i].x +',
        '            field[i].y*field[i].y);',
        '',
        '        if (amp > FP_1)',
        '        {',
        '            out[i].x = FP_1;',
        '            out[i].y = FP_0;',
        '        }',
        '        else',
        '        {',
        '            out[i] = field[i];',
        '        }',
        '    };',
        '',
        '__kernel void constrain_hologram(',
        '    __global {{ T.float }}2 const *field,',
        '    __global {{ T.float }}2 *out,',
        '    __global {{ T.float }} const *amplitude)',
        '    {',
        '        size_t i = get_global_id(0);',
        '        {{ T.float }} c, s;',
        '',
        '        {{ T.float }} phi = atan2(field[i].y, field[i].x);',
        '        s = sincos(phi, &c);',
        '',
        '        out[i].x = amplitude[i] * c;',
        '        out[i].y = amplitude[i] * s;',
        '    };',
    ])

    def __init__(self, spherical_wave: sw.SphericalWaveScalar, amplitude: np.ndarray, 
        phase: np.ndarray, pixelsize: float, wavelength: float, 
        dtype: np.dtype = np.complex64, default_fft_shift: bool = True,
        NA: float = 1.0, n_med: float = 1.0):

        '''
        Constructs a scalar electric field propagator that utilizes
        angular spectrum for propagation of light.

        Parameters
        ----------
        src_position: Tuple[float]
            Position of the source (m).
        amplitude: np.ndarray
            Real transmittance of the object.
        phase: np.ndarray
            Phase delay of the object exp(1j*phase)
        pixelsize: float
            Pixel size of the propagated object (m).
        wavelength: float
            Wavelength of light (m).
        dtype: np.dtype
            Type of the complex data array (hologram). Must be
            complex64 for single-precision or
            complex128 for double-precision.
        default_fft_shift: bool
            If True, the spatial frequencies are shifted so that the
            FFT of field/hologram does not have to be shifted. If False,
            the hologram is shifted after applying FFT and before applying
            inverse FFT.
        NA: float
            Numerical aperture of the imaging system.
        n_med: float
            Refractive index of the medium.
        '''

        self._shape = amplitude.shape
        self._x = pixelsize * np.arange(
            -(self._shape[1]-1)/2, (self._shape[1]-1)/2+1)
        self._y = pixelsize * np.arange(
            -(self._shape[0]-1)/2, (self._shape[0]-1)/2+1)

        self.spherical_wave_0 = spherical_wave.field(
            self._x[np.newaxis,:], self._y[:,np.newaxis], z=0.0)
        self._sw = spherical_wave

        self._pixelsize = float(pixelsize)
        self._wavelength = float(wavelength)
        self._cond_1_na = 1 - (NA/n_med)**2
        self._size = int(np.prod(amplitude.shape))
        self._dtype = np.dtype(dtype)
        self._fp_dtype = {
            np.complex64: np.dtype(np.float32),
            np.complex128: np.dtype(np.float64)}.get(self._dtype.type)
        if self._fp_dtype is None:
            raise TypeError('Expected dtype np.complex64 or np.complex128 '
                            'but got {}!'.format(dtype))

        # arrays for complex data field
        self._orig_amplitude = amplitude
        self._orig_phase = phase

        self._field = np.array(amplitude * np.exp(1j*phase), dtype=self._dtype)
        self._amplitude = np.array(amplitude, dtype=self._fp_dtype)

        # arrays of spatial frequencies (shifted to match FFT output)
        fx = np.array(np.fft.fftfreq(self._field.shape[1], self._pixelsize), dtype=self._fp_dtype)
        fy = np.array(np.fft.fftfreq(self._field.shape[0], self._pixelsize), dtype=self._fp_dtype)
        if default_fft_shift:
            self._fx, self._fy = np.meshgrid(
                fx, fy
            )
        else:
            self._fx, self._fy = np.meshgrid(
                np.fft.fftshift(fx), np.fft.fftshift(fy)
            )

        # by default the usage of devices (gpus) is turned off
        self._gpu = False
        self._default_fft_shift = default_fft_shift
        
        self._xcoors, self._ycoors = self._coordinates()
        self._backpropagate_sequential = False
        self._F_field = None

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

        # preparing reikna/cluda OpenCL kernels
        self._cluda_api = cluda.ocl_api()
        self._reikna_thread = self._cluda_api.Thread(self._cmdqueue)
        # creating kernel for computing FFT and inverse FFT
        self._cluda_field = self._reikna_thread.empty_like(self._field)
        self._cluda_field_fft = self._reikna_thread.empty_like(self._field)
        self._reikna_fft = clfft.FFT(self._field).compile(
            self._reikna_thread, compiler_options=clbuild_options)
        # creating kernel for shifting the FFT of complex field
        self._reikna_fft_shift = clfft.FFTShift(self._field).compile(
            self._reikna_thread, compiler_options=clbuild_options)

        # rendering OpenCL code from the jinja template
        # creating context data - just a single parameter T.float
        render_ctx = {
            np.complex64: {'T':{'float': 'float', 'cond_1_na': self._cond_1_na}},
            np.complex128: {'T':{'float': 'double', 'cond_1_na': self._cond_1_na}}
        }.get(self._dtype.type)
        # creating and rendering the template from the source code
        self._cl_source = jinja2.Template(
            self.OPENCL_CODE).render(**render_ctx)
        # building/compiling the OpenCL source code
        self._program = cl.Program(self._ctx, self._cl_source).build(
            options=clbuild_options)

        mf = cl.mem_flags

        # creating related OpenCL buffers
        self._cl_fx = cl.Buffer(self._ctx,
                                mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=self._fx)
        self._cl_fy = cl.Buffer(self._ctx,
                                mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=self._fy)

        # hologram intensity - amplitude
        self._cl_amplitude = cl.Buffer(self._ctx,
                                mf.READ_WRITE | mf.COPY_HOST_PTR,
                                hostbuf=self._amplitude)

        # scalar arguments of the propagation kernel
        self._np_wavelength = np.array((self._wavelength,), dtype=self._fp_dtype)
        self._np_z = np.zeros((1,), dtype=self._fp_dtype)

        self._np_spherical_wave_0 = np.empty((*self._shape, 2), dtype=self._fp_dtype)
        self._np_spherical_wave_0[..., 0] = self.spherical_wave_0.real
        self._np_spherical_wave_0[..., 1] = self.spherical_wave_0.imag

        self._np_spherical_wave_z = np.empty((*self._shape, 2), dtype=self._fp_dtype)

        self._np_div_amp_0 = np.array(1.0/np.abs(self.spherical_wave_0)**2, dtype=self._fp_dtype)

        self._cl_wavelength = cl.Buffer(self._ctx,
                                        mf.READ_WRITE | mf.COPY_HOST_PTR,
                                        hostbuf=self._np_wavelength)
        self._cl_z = cl.Buffer(self._ctx,
                               mf.READ_WRITE | mf.COPY_HOST_PTR,
                               hostbuf=self._np_z)
        self._cl_spherical_wave_0 = cl.Buffer(self._ctx,
                                mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self._np_spherical_wave_0)
        self._cl_spherical_wave_z = cl.Buffer(self._ctx,
                                mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self._np_spherical_wave_z)
        self._cl_div_amp_0 =  cl.Buffer(self._ctx,
                                mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self._np_div_amp_0)
        
        self._pre_propagation_sw_call = self._program.pre_propagation_sw
        self._pre_propagation_sw_call.set_args(
            self._cl_spherical_wave_0,
            self._cluda_field.base_data,
            self._cluda_field.base_data)
        
        self._field_propagate_call = self._program.field_propagate
        self._field_propagate_call.set_args(
            self._cluda_field_fft.base_data,
            self._cluda_field_fft.base_data,
            self._cl_fx,
            self._cl_fy,
            self._cl_z,
            self._cl_wavelength)
        
        self._post_propagation_sw_call = self._program.post_propagation_sw
        self._post_propagation_sw_call.set_args(
            self._cl_spherical_wave_z,
            self._cluda_field.base_data,
            self._cluda_field.base_data)
        
        self._pre_backpropagation_sw_call = self._program.pre_backpropagation_sw
        self._pre_backpropagation_sw_call.set_args(
            self._cl_spherical_wave_z,
            self._cluda_field.base_data,
            self._cluda_field.base_data)

        self._field_backpropagate_call = self._program.field_backpropagate
        self._field_backpropagate_call.set_args(
            self._cluda_field_fft.base_data,
            self._cluda_field_fft.base_data,
            self._cl_fx,
            self._cl_fy,
            self._cl_z,
            self._cl_wavelength)
        
        self._post_backpropagation_sw_call = self._program.post_backpropagation_sw
        self._post_backpropagation_sw_call.set_args(
            self._cl_spherical_wave_0,
            self._cl_div_amp_0,
            self._cluda_field.base_data,
            self._cluda_field.base_data)

        # preparing the absorption nonnegativity constrain for object
        self._constrain_nonneg_call = self._program.constrain_nonneg
        self._constrain_nonneg_call.set_args(
            self._cluda_field.base_data,
            self._cluda_field.base_data)

        # preparing the hologram constrain for image plane
        self._constrain_hologram_call = self._program.constrain_hologram
        self._constrain_hologram_call.set_args(
            self._cluda_field.base_data,
            self._cluda_field.base_data,
            self._cl_amplitude)

        # host to device transfer of amplitude, field, frequencies, z and wavelength
        cl.enqueue_copy(self._cmdqueue, self._cl_amplitude, self._amplitude)
        cl.enqueue_copy(self._cmdqueue, self._cluda_field.base_data, self._field)
        cl.enqueue_copy(self._cmdqueue, self._cl_fx, self._fx)
        cl.enqueue_copy(self._cmdqueue, self._cl_fy, self._fy)
        cl.enqueue_copy(self._cmdqueue, self._cl_z, self._np_z)
        cl.enqueue_copy(self._cmdqueue, self._cl_wavelength, self._np_wavelength)
        cl.enqueue_copy(self._cmdqueue, self._cl_spherical_wave_0, self._np_spherical_wave_0)

    def reset(self):
        self.field = np.array(self._orig_amplitude * np.exp(1j*self._orig_phase), dtype=self._dtype)

    @property
    def wavelength(self) -> float:

        '''
        Wavelength (in m)
        '''

        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value: float):

        self._wavelength = float(value)

        if self._gpu:
            self._np_wavelength = np.array((self._wavelength,), dtype=self._fp_dtype)
            cl.enqueue_copy(self._cmdqueue, self._cl_wavelength, self._np_wavelength)

    @property
    def field(self) -> np.ndarray:

        '''
        Scalar electric field.
        '''

        if self._gpu:
            cl.enqueue_copy(self._cmdqueue, self._field, self._cluda_field.base_data)

        return self._field

    @field.setter
    def field(self, value: np.ndarray):

        if value.shape != self._field.shape:
            raise ValueError(
                'Input field is not the same ' + \
                'shape as allocated field'
            )
        else:
            self._field = value

        if self._gpu:
            cl.enqueue_copy(self._cmdqueue, self._cluda_field.base_data, self._field)

    @property
    def amplitude(self) -> np.ndarray:

        '''
        Amplitude of the field.
        '''

        return np.abs(self.field)

    @property
    def phase(self) -> np.ndarray:

        '''
        Phase of the field within the interval [-pi, pi]
        (unwrapping required)
        '''

        return np.angle(self.field)

    @property
    def X(self):

        '''
        Mesh of x coordinates.
        '''

        return self._xcoors

    @property
    def Y(self):

        '''
        Mesh of y coordinates.
        '''

        return self._ycoors

    def _coordinates(self) -> Tuple[np.ndarray, np.ndarray]:

        '''
        Pixel coordinate grids for the object and propagated fields.

        Returns
        -------
        out: tuple
            Tuple of x any y coordinate grids.
        '''

        xcoors, ycoors = np.meshgrid(
            self._pixelsize * \
                np.arange(-np.floor(self._field.shape[1]/2), 
                    np.ceil(self._field.shape[1]/2)),
            self._pixelsize * \
                np.arange(-np.floor(self._field.shape[0]/2), 
                    np.ceil(self._field.shape[0]/2)))

        return xcoors, ycoors

    def backpropagate(self, z: float):

        '''
        Backpropagates the field for a distance z.

        Parameters
        ----------
        z: float
            Distance by which field is backpropagated (m).

        '''

        if self._gpu:
            if self._np_z[0] != z:
                self._np_z[0] = z
                sw_z = self._sw.field(self._x[np.newaxis,:], self._y[:,np.newaxis], z)
                self._np_spherical_wave_z[..., 0] = sw_z.real
                self._np_spherical_wave_z[..., 1] = sw_z.imag
                cl.enqueue_copy(self._cmdqueue, self._cl_z, self._np_z)
                cl.enqueue_copy(self._cmdqueue, self._cl_spherical_wave_z, self._np_spherical_wave_z)
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._pre_backpropagation_sw_call,
                [self._size], None)
            self._fft2gpu()
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._field_backpropagate_call,
                [self._size], None)
            self._ifft2gpu()
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._post_backpropagation_sw_call,
                [self._size], None)
        else:
            field = self._field - self._sw.field(self._x[np.newaxis,:], self._y[:,np.newaxis], z)
            F_field = self._fft2(field)
            H = self._transfer_function(-z)
            F_field_H = F_field * H
            field = self._ifft2(F_field_H)/self.spherical_wave_0
            self._field = field + 1.0
    
    def propagate(self, z: float):

        '''
        Propagates the field for a distance z.

        Parameters
        ----------
        z: float
            Distance by which field is propagated (m).

        '''

        if self._gpu:
            if self._np_z[0] != z:
                self._np_z[0] = z
                sw_z = self._sw.field(self._x[np.newaxis,:], self._y[:,np.newaxis], z)
                self._np_spherical_wave_z[..., 0] = sw_z.real
                self._np_spherical_wave_z[..., 1] = sw_z.imag
                cl.enqueue_copy(self._cmdqueue, self._cl_z, self._np_z)
                cl.enqueue_copy(self._cmdqueue, self._cl_spherical_wave_z, self._np_spherical_wave_z)
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._pre_propagation_sw_call,
                [self._size], None)
            self._fft2gpu()
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._field_propagate_call,
                [self._size], None)
            self._ifft2gpu()
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._post_propagation_sw_call,
                [self._size], None)
        else:
            field = (self._field - 1.0) * self.spherical_wave_0
            F_field = self._fft2(field)
            H = self._transfer_function(z)
            F_field_H = F_field * H
            field = self._ifft2(F_field_H)
            field += self._sw.field(self._x[np.newaxis,:], self._y[:,np.newaxis], z)
            self._field = field

    def backpropagate_sequential(self, dz: np.ndarray):

        '''
        Sequentially backpropagates the field by a distance dz.
        First do a backpropagate

        Parameters
        ----------
        dz: np.ndarray
            Distances by which the field is sequentially
            backpropagated (m).

        '''

        if not self._backpropagate_sequential:
            if self._gpu:
                self._fft2gpu()
            else:
                self._F_field = self._fft2(self._field)

        self._backpropagate_sequential = True

        if self._gpu:
            self._np_z[0] = dz
            cl.enqueue_copy(self._cmdqueue,
                self._cl_z, self._np_z)
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._field_backpropagate_call,
                [self._size], None)
            self._ifft2gpu()
            if not self._default_fft_shift:
                self._reikna_fft_shift(self._cluda_field_fft, self._cluda_field_fft)
        else:
            H = self._transfer_function(-dz)
            self._F_field = self._F_field * H
            self._field = self._ifft2(self._F_field)
            if not self._default_fft_shift:
                self._F_field = np.fft.ifftshift(self._F_field)            

        return self.field

    def _transfer_function(self, z: float) -> np.ndarray:

        '''
        Calculates the transfer function for
        the angular spectrum. For backpropagation 
        use -z.

        Parameters
        ----------
        z: float
            Distance by which field is propagated (m).
        '''

        arg = 1 - self._wavelength**2 * (self._fx**2 + self._fy**2)
        H = np.exp(1j * 2 * np.pi * z / self._wavelength * np.sqrt(arg))
        H[arg < self._cond_1_na] = 0.0

        return H

    def apply_constrain_nonneg(self):

        '''
        Applies the absorption non-negativity
        constrain that can be utilized for
        iterative reconstruction.
        '''

        if self._gpu:
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._constrain_nonneg_call,
                [self._size], None)
        else:

            constrain_mask = self.amplitude > 1.0

            amplitude = np.array(self.amplitude)
            amplitude[constrain_mask] = 1.0

            phase = np.array(self.phase)
            phase[constrain_mask] = 0.0

            self.field = amplitude * np.exp(1j*phase)

    def apply_constrain_hologram(self):
        
        '''
        Applies the hologram constrain in the image
        plane.
        '''

        if self._gpu:
            cl.enqueue_nd_range_kernel(self._cmdqueue, self._constrain_hologram_call,
                [self._size], None)
        else:
            self.field = self._amplitude * np.exp(1j*self.phase)

    def _fft2gpu(self):
        self._reikna_fft(self._cluda_field_fft, self._cluda_field, 0)
        if not self._default_fft_shift:
            self._reikna_fft_shift(self._cluda_field_fft, self._cluda_field_fft)

    def _fft2(self, field):
        if self._default_fft_shift:
            return np.fft.fft2(field)
        else:
            return np.fft.fftshift(np.fft.fft2(field))

    def _ifft2gpu(self):
        if not self._default_fft_shift:
            self._reikna_fft_shift(self._cluda_field_fft, self._cluda_field_fft)
        self._reikna_fft(self._cluda_field, self._cluda_field_fft, 1)

    def _ifft2(self, F):
        if self._default_fft_shift:
            return np.fft.ifft2(F)
        else:
            return np.fft.ifft2(np.fft.ifftshift(F))

class Iterator:

    def __init__(self, z: float | Sequence[float], propagator: Propagator | Sequence[Propagator],
        record_iterations: bool = False, verbose: bool = False):

        '''
        Iterative reconstruction using a single or multiple propagators.
        Propagator contains the angular spectrum algorithm with 
        preloaded recorded hologram for backpropagation.

        Parameters
        ----------
        z: float or list or tuple
            Positive distance between the object and hologram. If a list
            or tuple, it implies that the hologram was recorded at two
            or more distances and therefore requires corresponding 
            list or tuple of propagators.
        propagator: Propagator or list or tuple
            Type Propagator instance containing angular spectrum algorithm
            and recorded hologram at distance z. For multiple recorded distances
            it should be a list or tuple of corresponding propagators.
        record_iterations: bool
            If True, records amplitude and phase (in the object plane)
            at each iteration in a list.
            This feature also enables the use of sequence_export, that exports
            the amplitude and phase as a dictionary to be used in imAnimate.
        verbose: bool
            If True, prints the relative error at each iteration.
        '''

        self.propagator = propagator
        self.verbose = verbose

        if isinstance(z, list | tuple):
            self.z = np.array(z)
            self._multiple = True

            if isinstance(propagator, Sequence[propagator]) and \
                len(propagator) == len(z):
                raise ValueError('Propagator must be a list or tuple of\n '
                                 'propagators with the same size as z.')

            self._amplitude = [propagator[i].amplitude for i in range(len(propagator))]

            if verbose:
                print("Backpropagation without any constraints \n"
                    "using multiple-distance holograms.")
        else:
            self.z = z
            self._multiple = False
            self._amplitude = propagator.amplitude

            if verbose:
                print("Backpropagation with constrains.")

        # set the initial iteration point
        self._i = 0
        self._record_iterations = record_iterations
        self._amplitude_sequence = list()
        self._phase_sequence = list()

    @property
    def i(self) -> int:

        '''
        Current iteration step
        '''

        return self._i
    
    @property
    def field(self) -> np.ndarray:

        '''
        Object scalar electric field.
        '''

        if self._multiple:
            propagator = copy.deepcopy(self.propagator[-1])
            propagator.propagate(-self.z[-1])
            return propagator.field
        else:
            return self.propagator.field


    @property
    def amplitude(self) -> np.ndarray:

        '''
        Object amplitude at the current iteration step
        '''

        if self._multiple:
            propagator = copy.deepcopy(self.propagator[-1])
            propagator.propagate(-self.z[-1])
            return np.abs(propagator.field)
        else:
            return np.abs(self.propagator.field)

    @property
    def phase(self) -> np.ndarray:
        
        '''
        Object phase at the current iteration step
        '''

        if self._multiple:
            propagator = copy.deepcopy(self.propagator[-1])
            propagator.propagate(-self.z[-1])
            return np.angle(propagator.field)
        else:
            return np.angle(self.propagator.field)

    def iterate(self, iter_num=1):

        '''
        Single recorded hologram:
        First iteration propagates the hologram to the object 
        by distance z. Subsequently, each iteration consists
        of propagation to the detector and back to the object, 
        inbetween applying the constrains for reconstruction.

        Multiple recorded holgorams:
        Last propagator in the sequence is used to propagate
        to other holograms, where the amplitude of the propagated
        field is replaced by the recorded hologram. One interation
        includes propagation back to original position.

        Parameters:
        -----------
        iter_num: int
            Number of iterations to perform.
        '''

        for _ in range(iter_num):
            if self._multiple:
                delta_z = self.z[1:] - self.z[:-1]

                for j, dz in enumerate(delta_z):
                    # use only the last propagator
                    self.propagator[-1].propagate(-dz)
                    self.propagator[-1].amplitude = self.propagator[len(self.z) - j - 2].amplitude
                self.propagator[-1].propagate(delta_z.sum())

                # output the relative error
                if self.verbose:
                    print('Relative error (a.u.): {:3f}'.format(
                        np.sum(np.abs(self.propagator.amplitude-self._amplitude))/\
                            np.sum(np.abs(self._amplitude))))

                self.propagator[-1].amplitude = self._amplitude[-1]

                if self._record_iterations:
                    propagator = copy.deepcopy(self.propagator[-1])
                    propagator.propagate(-self.z[-1])
                    self._amplitude_sequence.append(np.abs(propagator.field))
                    self._phase_sequence.append(np.angle(propagator.field))
                self._i += 1

            else:
                if self._i == 0:
                    if self.verbose:
                        print('Initial z: ', self.z, 'm')
                    self.propagator.backpropagate(self.z)
                else:
                    self.propagator.apply_constrain_nonneg()
                    if self.verbose:
                        fig, ax = pp.subplots(1, 2, figsize=(8, 4))
                        ax[0].imshow(np.abs(self.propagator.field), cmap='gray')
                        ax[0].set_title('Amplitude (after constrains)')
                        ax[1].imshow(np.angle(self.propagator.field), cmap='gray')
                        ax[1].set_title('Phase (after constrains)')
                    self.propagator.propagate(self.z)
                    if self.verbose:
                        fig, ax = pp.subplots(1, 2, figsize=(8, 4))
                        ax[0].imshow(np.abs(self.propagator.field), cmap='gray')
                        ax[0].set_title('Amplitude (image plane)')
                        ax[1].imshow(np.angle(self.propagator.field), cmap='gray')
                        ax[1].set_title('Phase (image plane)')
                    # output the relative error
                    if self.verbose:
                        print('Relative error (a.u.): {:3f}'.format(
                            np.sum(np.abs(self.propagator.amplitude-self._amplitude))/\
                                np.sum(np.abs(self._amplitude))))
                    self.propagator.apply_constrain_hologram()
                    if self.verbose:
                        fig, ax = pp.subplots(1, 2, figsize=(8, 4))
                        ax[0].imshow(np.abs(self.propagator.field), cmap='gray')
                        ax[0].set_title('Amplitude (image domain constrains)')
                        ax[1].imshow(np.angle(self.propagator.field), cmap='gray')
                        ax[1].set_title('Phase (image domain constrains)')
                    self.propagator.backpropagate(self.z)
                    if self.verbose:
                        fig, ax = pp.subplots(1, 2, figsize=(8, 4))
                        ax[0].imshow(np.abs(self.propagator.field), cmap='gray')
                        ax[0].set_title('Amplitude (object plane)')
                        ax[1].imshow(np.angle(self.propagator.field), cmap='gray')
                        ax[1].set_title('Phase (object plane)')
                        pp.show()
                
                if self._record_iterations:
                    self._amplitude_sequence.append(np.abs(self.propagator.field))
                    self._phase_sequence.append(np.angle(self.propagator.field))
                self._i += 1