from __future__ import annotations

from typing import Tuple, List
import ctypes
import os
import time

import numpy as np
import pyopencl as cl
from jinja2 import Environment, BaseLoader


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


class ClTypes:
    pass


class ClTypesSingle(ClTypes):
    c_fp = ctypes.c_float
    c_fp_name = 'float'
    np_fp = np.dtype('float32')
    np_cplx = np.dtype('complex64')

    c_int = ctypes.c_int32
    c_int_name = 'int'
    np_int = np.dtype('int32')

    c_uint = ctypes.c_uint32
    c_uint_name = 'unsigned int'
    np_uint = np.dtype('uint32')


class ClTypesDouble(ClTypes):
    c_fp = ctypes.c_double
    c_fp_name = 'double'
    np_fp = np.dtype('float64')
    np_cplx = np.dtype('complex128')

    c_int = ctypes.c_int32
    c_int_name = 'int'
    np_int = np.dtype('int32')

    c_uint = ctypes.c_uint32
    c_uint_name = 'unsigned int'
    np_uint = np.dtype('uint32')


class ClCodeTemplate:
    def xl_defines(self) -> List[Tuple[str, int | str | float]] | None:
        return None

    def cl_declarations(self) -> str | None:
        return None

    def cl_source(self) -> str | None:
        return None


class ClWorker:
    def __init__(self, cl_devices: str | cl.Device | List['cl.Device'] |
                 cl.Context | cl.CommandQueue | None = None,
                 cl_build_options: List[str] | None = None,
                 cl_types: ClTypes = ClTypesSingle, verbose: bool = False):
        '''
        Initialize and prepare an OpenCL worker.

        Parameters
        ----------
        cl_devices: str | cl.Device | List['cl.Device'] | cl.Context | cl.CommandQueue | None
            Opencl device, device list, opencl context or opencl queue.
        cl_types: ClTypes
            A set of data types that will be used by this worker instance.\
        verbose: bool
            Enables verbose output.
        '''
        self._cl_types = cl_types
        if cl_build_options is None:
            cl_build_options = []
        self._cl_build_options = cl_build_options
        self._queue = None

        self._verbose = bool(verbose)

        if cl_devices is None:
            cl_devices = cl.create_some_context()

        if isinstance(cl_devices, cl.Device):
            self._queue = cl.CommandQueue(cl.Context([cl_devices]))
        elif isinstance(cl_devices, (list, tuple)):
            self._queue = cl.CommandQueue(cl.Context(cl_devices))
        elif isinstance(cl_devices, cl.Context):
            self._queue = cl.CommandQueue(cl_devices)
        elif isinstance(cl_devices, cl.CommandQueue):
            self._queue = cl_devices

    @property
    def cl_queue(self) -> cl.CommandQueue | None:
        return self._queue

    @property
    def cl_context(self) -> cl.Context | None:
        if self._queue is not None:
            return self._queue.context
        return None

    @property
    def cl_devices(self) -> List[cl.Device] | None:
        if self._queue is not None:
            return self._queue.context.devices
        return None

    @property
    def is_valid(self) -> bool:
        return self._queue is not None

    @property
    def types(self) -> ClTypes:
        return self._cl_types

    @property
    def cl_build_options(self) -> List[str]:
        return self._cl_build_options

    @property
    def verbose(self) -> bool:
        return self._verbose


CL_TEMPLATE_COMMON = '\n'.join([
    '{% if T.c_fp_name == "float" %}',
    '#define FP(value)   value##f',
    '{%- else -%}',
    '#define FP(value)   value',
    '{%- endif %}',
    '',
    '#define PI          FP(3.141592653589793)',
    '#define FP_0        FP(0.0)',
    '#define FP_1        FP(1.0)',
    '#define FP_2        FP(2.0)',
    '#define FP_9        FP(9.0)',
])

CL_RAY_SOM_TEMPLATE = '\n'.join([
    '__kernel void kernel_propagate('
    '    __global {{ T.c_fp_name }}2 const *field,',
    '    __global {{ T.c_fp_name }}2 *out,',
    '    __global {{ T.c_fp_name }} const *k,',
    '    __global {{ T.c_fp_name }} const *xf,',
    '    __global {{ T.c_fp_name }} const *yf,',
    '    __global {{ T.c_fp_name }} const *zf,',
    '    __global {{ T.c_uint_name }} const *width_f,',
    '    __global {{ T.c_uint_name }} const *height_f,',
    '    __global {{ T.c_fp_name }} const *dxf,',
    '    __global {{ T.c_fp_name }} const *dyf,',
    '    __global {{ T.c_fp_name }} const *xo,',
    '    __global {{ T.c_fp_name }} const *yo,',
    '    __global {{ T.c_fp_name }} const *zo)',
    '{',
    '    size_t i = get_global_id(0);',
    '    {{ T.c_fp_name }}2 q = ({{ T.c_fp_name }}2)(FP_0, FP_0);',
    '    {{ T.c_fp_name }} x = xo[i];',
    '    {{ T.c_fp_name }} y = yo[i];',
    '    {{ T.c_fp_name }} z = zo[i];',
    '    {{ T.c_fp_name }} dz = z - (*zf);',
    '    {{ T.c_fp_name }} dz_dz = dz * dz;',
    '    size_t flat_if = 0;',
    '    size_t input_width = *width_f;',
    '    size_t input_height = *height_f;',
    '',
    '    for (size_t i_yf = 0; i_yf < input_height; ++i_yf) {',
    '        for (size_t i_xf = 0; i_xf < input_width; ++i_xf) {',
    '            if (field[flat_if].x != FP_0 || field[flat_if].y != FP_0) {',
    '                {{ T.c_fp_name }} dx = x - xf[i_xf];',
    '                {{ T.c_fp_name }} dy = y - yf[i_yf];',
    '                {{ T.c_fp_name }} r_r = dx*dx + dy*dy + dz_dz;',
    '                {{ T.c_fp_name }} r = sqrt(r_r);',
    '                {{ T.c_fp_name }} inv_r = FP_1 / r;',
    '                {{ T.c_fp_name }} amp = z / (FP_2 * PI * r_r);',
    '                {{ T.c_fp_name }} s, c;',
    '                s = sincos((*k) * r, &c);',
    '                {{ T.c_fp_name }}2 t1 = ({{ T.c_fp_name }}2)(amp * c, amp * s);',
    '                {{ T.c_fp_name }}2 t2 = ({{ T.c_fp_name }}2)(inv_r, -(*k));',
    '                {{ T.c_fp_name }}2 tmp;',
    '                tmp.x = t1.x * t2.x - t1.y * t2.y;',
    '                tmp.y = t1.x * t2.y + t1.y * t2.x;',
    '',
    '                // size_t flat_if = i_xf + i_yf*input_width;',
    '                q.x += tmp.x * field[flat_if].x - tmp.y * field[flat_if].y;',
    '                q.y += tmp.x * field[flat_if].y + tmp.y * field[flat_if].x;',
    '            }',
    '            ++flat_if;',
    '        }',
    '    }',
    '    {{ T.c_fp_name }} da = ((*dxf) * (*dyf));',
    '    out[i].x = q.x * da;',
    '    out[i].y = q.y * da;',
    '}',
    '',
    '__kernel void kernel_propagate_simpson('
    '    __global {{ T.c_fp_name }}2 const *field,',
    '    __global {{ T.c_fp_name }}2 *out,',
    '    __global {{ T.c_fp_name }} const *k,',
    '    __global {{ T.c_fp_name }} const *xf,',
    '    __global {{ T.c_fp_name }} const *yf,',
    '    __global {{ T.c_fp_name }} const *zf,',
    '    __global {{ T.c_uint_name }} const *width_f,',
    '    __global {{ T.c_uint_name }} const *height_f,',
    '    __global {{ T.c_fp_name }} const *dxf,',
    '    __global {{ T.c_fp_name }} const *dyf,',
    '    __global {{ T.c_fp_name }} const *xo,',
    '    __global {{ T.c_fp_name }} const *yo,',
    '    __global {{ T.c_fp_name }} const *zo)',
    '{',
    '    size_t i = get_global_id(0);',
    '    {{ T.c_fp_name }}2 q = ({{ T.c_fp_name }}2)(FP_0, FP_0);',
    '    {{ T.c_fp_name }} x = xo[i];',
    '    {{ T.c_fp_name }} y = yo[i];',
    '    {{ T.c_fp_name }} z = zo[i];',
    '    {{ T.c_fp_name }} dz = z - (*zf);',
    '    {{ T.c_fp_name }} dz_dz = dz * dz;',
    '    size_t flat_if = 0;',
    '    size_t input_width = *width_f;',
    '    size_t input_height = *height_f;',
    '',
    '    for (size_t i_yf = 0; i_yf < input_height; ++i_yf) {',
    '        for (size_t i_xf = 0; i_xf < input_width; ++i_xf) {',
    '            if (field[flat_if].x != FP_0 || field[flat_if].y != FP_0) {',
    '                int wx = (i_xf & 1) ? 4 : (i_xf == 0 || i_xf == input_width - 1) ? 1 : 2;',
    '                int wy = (i_yf & 1) ? 4 : (i_yf == 0 || i_yf == input_height - 1) ? 1 : 2;',
    '                {{ T.c_fp_name }} w = wx*wy;',
    '',
    '                {{ T.c_fp_name }} dx = x - xf[i_xf];',
    '                {{ T.c_fp_name }} dy = y - yf[i_yf];',
    '                {{ T.c_fp_name }} r_r = dx*dx + dy*dy + dz_dz;',
    '                {{ T.c_fp_name }} r = sqrt(r_r);',
    '                {{ T.c_fp_name }} inv_r = FP_1 / r;',
    '                {{ T.c_fp_name }} amp = z * w / (FP_2 * PI * r_r);',
    '                {{ T.c_fp_name }} s, c;',
    '                s = sincos((*k) * r, &c);',
    '                {{ T.c_fp_name }}2 t1 = ({{ T.c_fp_name }}2)(amp * c, amp * s);',
    '                {{ T.c_fp_name }}2 t2 = ({{ T.c_fp_name }}2)(inv_r, -(*k));',
    '                {{ T.c_fp_name }}2 tmp;',
    '                tmp.x = t1.x * t2.x - t1.y * t2.y;',
    '                tmp.y = t1.x * t2.y + t1.y * t2.x;',
    '',
    '                // size_t flat_if = i_xf + i_yf*input_width;',
    '                q.x += tmp.x * field[flat_if].x - tmp.y * field[flat_if].y;',
    '                q.y += tmp.x * field[flat_if].y + tmp.y * field[flat_if].x;',
    '            }',
    '            ++flat_if;',
    '        }',
    '    }',
    '    {{ T.c_fp_name }} da = ((*dxf) * (*dyf)) / FP_9;',
    '    out[i].x = q.x * da;',
    '    out[i].y = q.y * da;',
    '}',
])


class RaySomPix(ClWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cl_field = self._cl_result = None
        self._cl_field_x = self._cl_field_y = None
        self._cl_field_z = None
        self._cl_field_dx = self._cl_field_dy = None
        self._cl_k = None

        self._cl_x = self._cl_y = self._cl_z = None
        self._np_x = self._np_y = self._np_z = None
        self._np_field_dx = self._np_field_dy = None

        self._kernel_propagate_call = None
        self._pipeline = None

        # building the OpenCL program
        t = time.perf_counter()
        self._program = cl.Program(self.cl_context, self.cl_source()).build(
            options=self.cl_build_options)
        dt = time.perf_counter() - t
        if self.verbose:
            print('Kernel built in {:.6f} s'.format(dt))

    def cl_source(self) -> str:
        '''
        Returns OpenCL source code ready to be compiled.
        '''
        src = '\n'.join([CL_TEMPLATE_COMMON, '', CL_RAY_SOM_TEMPLATE])
        src_template = Environment(loader=BaseLoader).from_string(src)
        return src_template.render(T=self.types)

    def initialize(self, field: np.ndarray,
                   pixelsize: float | Tuple[float, float],
                   wavelength: float, n: float = 1.0, z: float = 0.0):
        '''
        Initializes the numerical engine.

        Parameters
        ----------
        field: np.ndarray
            Complex field to propagate defined as amplitude*exp(1j*phase).
            Origin (0, 0, z) of the coordinate system is pinned
            to the geometric center of the input field.
        pixelsize: float | Tuple[float, float]
            Size of the pixel (m) as a single float value for square pixels
            or a tuple (dy, dx) for rectangular pixels.
        wavelength: float
            Wavelength of light (m) in vacuum.
        n: float
            Refractive index of the medium.
        z: float
            Z coordinate of the provided complex field.
        '''
        if isinstance(pixelsize, (float ,int)):
            pixelsize = float(pixelsize)
            pixelsize = (pixelsize, pixelsize)
        else:
            pixelsize = (float(pixelsize[0]), float(pixelsize[1]))

        self._np_field = np.array(field, dtype=self.types.np_cplx)

        dy, dx = pixelsize
        height, width = self._np_field.shape

        self._np_field_x = np.arange(
            -0.5*(width - 1)*dx, 0.5*width*dx, dx, dtype=self.types.np_fp)
        self._np_field_y = np.arange(
            -0.5*(height - 1)*dy, 0.5*height*dy, dy, dtype=self.types.np_fp)
        self._np_field_z = np.array([float(z)], dtype=self.types.np_fp)
        self._np_field_dx = np.array([dx], dtype=self.types.np_fp)
        self._np_field_dy = np.array([dy], dtype=self.types.np_fp)
        self._np_field_y_grid, self._np_field_x_grid = np.meshgrid(
            self._np_field_y, self._np_field_x, indexing='ij')

        self._wavelength = float(wavelength)
        self._n = float(n)
        self._k = 2.0*np.pi*self._n/self._wavelength
        self._np_k = np.array([self._k], dtype=self.types.np_fp)

        self._np_field_width = np.array([width], dtype=self.types.np_uint)
        self._np_field_height = np.array([height], dtype=self.types.np_uint)

        self._prepare_field_cl()

        self._pipeline = True

    @property
    def x(self) -> np.ndarray:
        ''' Returns a vector of field coordinates along the x axis (m). '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        return self._np_field_x

    @property
    def y(self) -> np.ndarray:
        ''' Returns a vector of field coordinates y along the y axis (m). '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        return self._np_field_y

    @property
    def x_grid(self) -> np.ndarray:
        ''' Returns a 2D array of field coordinates x (m). '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        return self._np_field_x_grid

    @property
    def y_grid(self) -> np.ndarray:
        ''' Returns a 2D array of field coordinates y (m). '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        return self._np_field_y_grid

    @property
    def dx(self) -> np.ndarray:
        ''' Returns pixel size x (m). '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        return self._np_field_dx[0]

    @property
    def dy(self) -> np.ndarray:
        ''' Return pixel size y (m). '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        return self._np_field_dy[0]

    def get_field(self, out: np.ndarray | None = None) -> np.ndarray:
        '''
        Returns the current input field as passed to the
        :py:meth:`~RaySomPix.initialize` method.

        Parameters
        ----------
        out: np.ndarray | None
            Optional output array for the complex field.

        Returns
        -------
        field: np.ndarray
            Field as a complex numpy array.
        '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')

        if out is not None:
            if out.shape != self._np_field.shape:
                raise ValueError('Shape of the output array does not match '
                                 'the shape of the complex field!')
            if out.dtype != self.types.np_cplx:
                raise TypeError('Type of the output array does not match '
                                'the type of the complex field!')
        else:
            out = np.empty(self._np_field.shape, self.types.np_cplx)

        cl.enqueue_copy(self.cl_queue, out, self._cl_field)

        return out

    def set_field(self, field: np.ndarray) -> np.ndarray:
        '''
        Set the complex field to the specified array.

        Parameters
        ----------
        field: np.ndarray
            Complex field.
        '''
        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')
        if field.shape != self._np_field.shape:
            raise ValueError('Shape of the input array does not match '
                             'the shape of the complex field!')

        np.copyto(self._np_field, field)
        cl.enqueue_copy(self.cl_queue, self._cl_field, self._np_field)

    field = property(get_field, set_field, None, 'Complex field.')

    def _prepare_field_cl(self):
        '''
        Allocate buffers for the input field, related coordinate system,
        pixel size, and wavenumber.
        '''
        mf = cl.mem_flags

        self._cl_field = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field)

        self._cl_field_x = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_x)

        self._cl_field_y = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_y)

        self._cl_field_z = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_z)

        self._cl_field_width = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_width)

        self._cl_field_height = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_height)

        self._cl_field_dx = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_dx)

        self._cl_field_dy = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_field_dy)

        self._cl_k = cl.Buffer(
            self.cl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self._np_k)


    def _prepare_propagate_cl(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              k: float):
        '''
        Prepare the OpenCL buffers and kernel for execution.

        Parameters
        ----------
        x: np.ndarray
            X coordinates of the points at which to compute the field.
        y: np.ndarray
            Y coordinates of the points at which to compute the field.
        z: np.ndarray
            Z coordinates of the points at which to compute the field.
        k: float
            Wavenumber in the medium defined as :math:`2\\pi/(\\lambda/n)`.
        '''
        rebind = False

        n = max(x.size, y.size, z.size)

        np_cplx_itemsize = np.dtype(self.types.np_cplx).itemsize
        np_fp_itemsize = np.dtype(self.types.np_fp).itemsize

        if self._np_x is None or self._np_x.size < x.size:
            self._np_x = np.array(x.flat, dtype=self.types.np_fp)
        if self._np_y is None or self._np_y.size < y.size:
            self._np_y = np.array(y.flat, dtype=self.types.np_fp)
        if self._np_z is None or self._np_z.size < z.size:
            self._np_z = np.array(z.flat, dtype=self.types.np_fp)

        if self._cl_x is None or self._cl_x.size < n*np_fp_itemsize:
            self._cl_x = cl.Buffer(
                self.cl_context, cl.mem_flags.READ_ONLY, n*np_fp_itemsize)
            rebind = True
        if self._cl_y is None or self._cl_y.size < n*np_fp_itemsize:
            self._cl_y = cl.Buffer(
                self.cl_context, cl.mem_flags.READ_ONLY, n*np_fp_itemsize)
            rebind = True
        if self._cl_z is None or self._cl_z.size < n*np_fp_itemsize:
            self._cl_z = cl.Buffer(
                self.cl_context, cl.mem_flags.READ_ONLY, n*np_fp_itemsize)
            rebind = True

        if self._cl_result is None or self._cl_result.size < x.size*np_cplx_itemsize:
            self._cl_result = cl.Buffer(
                self.cl_context, cl.mem_flags.READ_WRITE,
                x.size*np_cplx_itemsize)

        np.copyto(self._np_x[:n], x.flat)
        np.copyto(self._np_y[:n], y.flat)
        np.copyto(self._np_z[:n], z.flat)

        cl.enqueue_copy(self.cl_queue, self._cl_x, self._np_x[:n])
        cl.enqueue_copy(self.cl_queue, self._cl_y, self._np_y[:n])
        cl.enqueue_copy(self.cl_queue, self._cl_z, self._np_z[:n])

        if rebind or self._kernel_propagate_call is None:
            self._kernel_propagate_call = self._program.kernel_propagate
            self._kernel_propagate_call.set_args(
                self._cl_field,
                self._cl_result,
                self._cl_k,
                self._cl_field_x,
                self._cl_field_y,
                self._cl_field_z,
                self._cl_field_width,
                self._cl_field_height,
                self._cl_field_dx,
                self._cl_field_dy,
                self._cl_x,
                self._cl_y,
                self._cl_z)

            self._kernel_propagate_simpson_call = \
                self._program.kernel_propagate_simpson
            self._kernel_propagate_simpson_call.set_args(
                self._cl_field,
                self._cl_result,
                self._cl_k,
                self._cl_field_x,
                self._cl_field_y,
                self._cl_field_z,
                self._cl_field_width,
                self._cl_field_height,
                self._cl_field_dx,
                self._cl_field_dy,
                self._cl_x,
                self._cl_y,
                self._cl_z)
            
        if k != self._k:
            self._np_k[0] = k
            self._k = k
            cl.enqueue_copy(self.cl_queue, self._cl_k, self._np_k)

    def propagate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  wavelength: float | None = None,
                  n: float | None = None, simpson: bool = False,
                  out: np.ndarray | None = None,
                  verbose: bool = False) -> np.ndarray:
        '''
        Compute field at points defined by arrays x, y and z. Note that
        the shape of x, y and z arrays must match.

        Parameters
        ----------
        x: np.ndarray
            A numpy array of x coordinates at which to compute the field.
        y: np.ndarray
            A numpy array of y coordinates at which to compute the field.
        z: np.ndarray
            A numpy array of z coordinates at which to compute the field.
        n: float | None
            Refractive index of the medium or None to use the refractive index
            passed to the initialize method.
        wavelength: float | None
            Wavelength of light (m) or None to use the wavelength passed to
            the initialize method.
        simpson: bool
            Use 2D Simpson method to compute the Rayleigh-Sommerfeld integral.
            Note that the number of elements along the x and y axis of the
            input field must be odd!
        out: np.ndarray | None
            Placeholder for the computed complex field. Note that the type
            of the array must match the type of the complex numbers in the
            OpenCL kernel, and the shape of the array must match the shape
            of the x, y and z arrays.
        verbose: bool
            Enables verbose output.

        Returns
        -------
        field: np.ndarray
            Placeholder for the computed complex field. Note that the type
            of the array must match the type of the complex numbers in the
            OpenCL kernel, and the shape of the array must match the shape
            of the x, y and z arrays.

        Note
        ----
        Geometric center of the input field passed to the 
        :py:meth:`~RaySomPix.initialize` is pinned to the geometric center
        of the field.
        '''
        t_1 = time.perf_counter()

        if self._pipeline is None:
            raise RuntimeError('Propagator pipeline is not initialized!')

        if self._np_field is None:
            raise ValueError('Kernel is uninitialized!')

        if x.shape != y.shape or y.shape != z.shape:
            raise ValueError('The shape of x, y and z must be the same!')

        if out is None:
            out = np.empty(x.shape, dtype=self.types.np_cplx)
        else:
            if out.shape != x.shape:
                raise ValueError('Shape of the output array must be the same '
                                 'as the shape of the coordinates arrays '
                                 'x, y and z!')
            if out.dtype != self.types.np_cplx:
                raise TypeError('Type of the output array must match the '
                                'complex floating-point type of the kernel!')

        if wavelength is None:
            wavelength = self._wavelength
        if n is None:
            n = self._n

        k = np.pi*2*n/wavelength

        self._prepare_propagate_cl(x, y, z, k)

        t_2 = time.perf_counter()

        if simpson:
            w, h = self._np_field_width[0], self._np_field_height[0]
            if w < 3 or h < 3:
                raise ValueError(
                    'Simpson integral requires at least 3 elements '
                    'along the x and y axis of the input field!')
            if w & 1 == 0 or h & 1  == 0:
                raise ValueError(
                    'Simpson integral requires odd number of elements '
                    'along the x and y axis of the input field!')

            cl.enqueue_nd_range_kernel(
                self.cl_queue, self._kernel_propagate_simpson_call,
                [x.size], None
            ).wait()

        else:
            cl.enqueue_nd_range_kernel(
                self.cl_queue, self._kernel_propagate_call,
                [x.size], None
            ).wait()

        t_3 = time.perf_counter()

        cl.enqueue_copy(self.cl_queue, out, self._cl_result).wait()

        t_4 = time.perf_counter()

        if self.verbose or verbose:
            meth = 'simpson' if simpson else 'basic'
            print('RaySomPix.propagate in {} points (method = {}):'.format(
                x.size, meth))
            print('  Prepare : {:.3f} s'.format(t_2 - t_1))
            print('  Compute : {:.3f} s'.format(t_3 - t_2))
            print('  Download: {:.3f} s'.format(t_4 - t_3))

        return out


if __name__ == '__main__':
    width = height = 201
    dx = dy = 0.05e-6

    x = np.arange(-dx*(width - 1)*0.5, dx*width*0.5, dx)
    y = np.arange(-dy*(height - 1)*0.5, dy*height*0.5, dy)
    Y, X = np.meshgrid(y, x, indexing='ij')

    amplitude = np.zeros((height, width))
    amplitude[X**2 + Y**2 <= (2.5e-6)**2] = 1.0
    phase = np.zeros_like(amplitude)

    field = amplitude*np.exp(1j*phase)

    p = RaySomPix(cl.create_some_context(False),
                  cl_types=ClTypesSingle,
                  cl_build_options=['-cl-mad-enable'], verbose=True)
    p.initialize(amplitude*np.exp(1j*phase), pixelsize=(dy, dx),
                 wavelength=550e-9, n=1.0, z=0.0)

    dxi = dyi = 0.05e-6
    zi = 10e-6
    widthi = heighti = 400
    xi = np.arange(-dxi*(widthi - 1)*0.5, dxi*widthi*0.5, dxi)
    yi = np.arange(-dyi*(heighti - 1)*0.5, dyi*heighti*0.5, dyi)
    Yi, Xi = np.meshgrid(yi, xi, indexing='ij')
    Zi = np.tile(zi, Xi.shape)

    result = p.propagate(Xi, Yi, Zi)
    result_s = p.propagate(Xi, Yi, Zi, simpson=True)

    import matplotlib.pyplot as pp
    extent = [x.min(), x.max(), y.max(), y.min()]
    fig, ax = pp.subplots(2, 2)
    im_f = ax[0 ,0].imshow(np.abs(field), extent=extent, origin='upper')
    ax[0, 0].set_title('Field @ z=0')
    fig.colorbar(im_f, ax=ax[0, 0])
    im_rs = ax[1, 0].imshow(np.abs(result), extent=extent, origin='upper')
    ax[1, 0].set_title('Basic: Field @ z={:.6f} mm'.format(zi*1e3))
    fig.colorbar(im_rs, ax=ax[1, 0])
    im_rsp = ax[1, 1].imshow(np.abs(result_s), extent=extent, origin='upper')
    ax[1, 1].set_title('Simpson: Field @ z={:.6f} mm'.format(zi*1e3))
    fig.colorbar(im_rsp, ax=ax[1, 1])
    im_err = ax[0, 1].imshow(np.abs(result - result_s)/np.abs(result_s)*100.0,
                             extent=extent, origin='upper')
    ax[0, 1].set_title('Rel. diff. (%) @ z={:.6f} mm'.format(zi*1e3))
    fig.colorbar(im_err, ax=ax[0, 1])
    pp.tight_layout()
    pp.show()