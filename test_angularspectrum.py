# this script compares angular spectrum method with
# direct Rayleigh-Sommerfeld integration method for
# for a simple case of a collections of semi-transparent
# circular objects
from time import perf_counter
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as pp

import clinfo
import raysom as rs
import angularspectrum as asp
import phantom as ph
import scalarwave as sw

vmin = -10
vmax = 10

vendor = 'AMD' # select the vendor of the GPU
datatype = np.complex128 # select the datatype for the calculations

params_cl = {
    'cldevices': [clinfo.gpus(vendor)[0],], # select the first GPU
    'clbuild_options': ['-cl-mad-enable', '-cl-fast-relaxed-math'], # build options
}

params = {
    'wavelength': 550e-9, # wavelength of the light [m]
    'pixelsize': 0.25e-6, # pixel size of the camera [m]
    'z': 100e-6, # distance from the camera to the object plane [m]
}

params_circle = {
    'pixelsize': params['pixelsize'], # pixel size of the object [m]
    'shape': (512, 512), # shape of the object
    'radii': [10e-6,], # radius of the circles [m]
    'centers': [[0e-6,  0e-6],], # centers of the circles [m]
    'amplitudes': [1.0,], # amplitudes of the circles
    'phases': [0.0, ], # phases of the circles
}

params_rectangle = {
    'pixelsize': params['pixelsize'], # pixel size of the object [m]
    'shape': (512, 512), # shape of the object
    'lx': [30e-6,], # width of the rectangle [m]
    'ly': [20e-6,], # height of the rectangle [m]
    'centers': [[0e-6,  0e-6],], # centers of the circles [m]
    'amplitudes': [1.0,], # amplitude of the rectangle
    'phases': [0.0, ], # phase of the rectangle
}

def plane_wave_aperture(aperture_type='circle'):

    if aperture_type == 'circle':
        aperture = ph.Circle(**params_circle, background=0.0)
    elif aperture_type == 'rectangle':
        aperture = ph.Rectangle(**params_rectangle, background=0.0)

    fig, ax = pp.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(aperture.amplitude, cmap='gray')
    ax[0].set_title('Amplitude')
    ax[1].imshow(aperture.phase, cmap='gray')
    ax[1].set_title('Phase')

    # EXAMPLE OF THE USE OF THE RAYLEIGH-SOMMERFELD PROPAGATOR
    aperture_refined = aperture.refine(2)

    field = aperture_refined.field
    propagator = rs.PropagatorCl(
        amplitude=np.abs(field),
        phase=np.angle(field),
        pixelsize=2*[aperture_refined.pixelsize,],
        wavelength=params['wavelength'],
        dtype=datatype
    )

    propagator.init_gpu(**params_cl)
    fun_field_z = lambda z: propagator.propagate(
        x=aperture.x, y=aperture.y, z=z
    )

    t0 = perf_counter()
    field_z_rs = fun_field_z(params['z'])
    print(f'Elapsed time (Rayleigh-Sommerfeld):', perf_counter() - t0)

    propagator = asp.Propagator(
        amplitude=aperture.amplitude - 1.0,  # Babinet's principle
        phase=aperture.phase,
        pixelsize=aperture.pixelsize,
        wavelength=params['wavelength'],
        dtype=datatype
    )    

    propagator.init_gpu(**params_cl)

    def fun_field_z(z):
        propagator.propagate(z)
        return propagator.field + sw.PlaneWaveScalar(k=2*np.pi/params['wavelength']).field(z)
    
    t0 = perf_counter()
    field_z_asp = fun_field_z(params['z'])
    print(f'Elapsed time (Angular spectrum):', perf_counter() - t0)

    # Amplitudes and relative differences
    fig, ax = pp.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.abs(field_z_rs), cmap='gray')
    ax[0].set_title(f'Amplitude {aperture_type} (Rayleigh-Sommerfeld)')
    ax[1].imshow(np.abs(field_z_asp), cmap='gray')
    ax[1].set_title(f'Amplitude {aperture_type} (Angular spectrum)')
    axi = ax[2].imshow((np.abs(field_z_rs)-np.abs(field_z_asp))/np.abs(field_z_asp),
        cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax[2].set_title('Relative difference (%)')
    fig.colorbar(axi, ax=ax[2])

    # Phases and relative differences
    fig, ax = pp.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.angle(field_z_rs), cmap='gray')
    ax[0].set_title(f'Phase {aperture_type} (Rayleigh-Sommerfeld)')
    ax[1].imshow(np.angle(field_z_asp), cmap='gray')
    ax[1].set_title(f'Phase {aperture_type} (Angular spectrum)')
    axi = ax[2].imshow(100*(np.angle(field_z_rs) - np.angle(field_z_asp))/np.angle(field_z_asp),
        cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax[2].set_title('Relative difference (%)')
    fig.colorbar(axi, ax=ax[2])

def plane_wave_aperture_negative(aperture_type='circle'):

    if aperture_type == 'circle':
        _params_circle = deepcopy(params_circle)
        _params_circle['amplitudes'] = [0.0,]
        aperture = ph.Circle(**_params_circle, background=1.0)
    elif aperture_type == 'rectangle':
        _params_rectangle = deepcopy(params_rectangle)
        _params_rectangle['amplitudes'] = [0.0,]
        aperture = ph.Rectangle(**_params_rectangle, background=1.0)

    fig, ax = pp.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(aperture.amplitude, cmap='gray')
    ax[0].set_title('Amplitude')
    ax[1].imshow(aperture.phase, cmap='gray')
    ax[1].set_title('Phase')

    # EXAMPLE OF THE USE OF THE RAYLEIGH-SOMMERFELD PROPAGATOR
    aperture_refined = aperture.refine(2)

    field = aperture_refined.field - sw.PlaneWaveScalar(k=2*np.pi/params['wavelength']).field(0) # Babinet's principle
    propagator = rs.PropagatorCl(
        amplitude=np.abs(field),
        phase=np.angle(field),
        pixelsize=2*[aperture_refined.pixelsize,],
        wavelength=params['wavelength'],
        dtype=datatype
    )

    propagator.init_gpu(**params_cl)
    fun_field_z = lambda z: propagator.propagate(
        x=aperture.x, y=aperture.y, z=z
    )

    t0 = perf_counter()
    field_z_rs = fun_field_z(params['z']) + sw.PlaneWaveScalar(k=2*np.pi/params['wavelength']).field(params['z'])
    print(f'Elapsed time (Rayleigh-Sommerfeld):', perf_counter() - t0)

    propagator = asp.Propagator(
        amplitude=aperture.amplitude,
        phase=aperture.phase,
        pixelsize=aperture.pixelsize,
        wavelength=params['wavelength'],
        dtype=datatype
    )    

    propagator.init_gpu(**params_cl)
    def fun_field_z(z):
        propagator.propagate(z)
        return propagator.field
    
    t0 = perf_counter()
    field_z_asp = fun_field_z(params['z'])
    print(f'Elapsed time (Angular spectrum):', perf_counter() - t0)

    # Amplitudes and relative differences
    fig, ax = pp.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.abs(field_z_rs), cmap='gray')
    ax[0].set_title(f'Amplitude {aperture_type} (Rayleigh-Sommerfeld)')
    ax[1].imshow(np.abs(field_z_asp), cmap='gray')
    ax[1].set_title(f'Amplitude {aperture_type} (Angular spectrum)')
    axi = ax[2].imshow((np.abs(field_z_rs)-np.abs(field_z_asp))/np.abs(field_z_asp),
        cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax[2].set_title('Relative difference (%)')
    fig.colorbar(axi, ax=ax[2])

    # Phases and relative differences
    fig, ax = pp.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.angle(field_z_rs), cmap='gray')
    ax[0].set_title(f'Phase {aperture_type} (Rayleigh-Sommerfeld)')
    ax[1].imshow(np.angle(field_z_asp), cmap='gray')
    ax[1].set_title(f'Phase {aperture_type} (Angular spectrum)')
    axi = ax[2].imshow(100*(np.angle(field_z_rs) - np.angle(field_z_asp))/np.angle(field_z_asp),
        cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax[2].set_title('Relative difference (%)')
    fig.colorbar(axi, ax=ax[2])

# plane_wave_aperture(aperture_type='circle')
# plane_wave_aperture(aperture_type='rectangle')
plane_wave_aperture_negative(aperture_type='circle')
# plane_wave_aperture_negative(aperture_type='rectangle')
pp.show()