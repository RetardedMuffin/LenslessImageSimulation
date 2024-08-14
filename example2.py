from time import perf_counter

import numpy as np
import matplotlib.pyplot as pp

import clinfo
import raysompix as rsp
import angularspectrum as asp
import scalarwave as sw

# definicija za grafično kartico
vendor = 'AMD' # del imena grafične kartice
datatype = np.complex128 # računska natančnost (ni potrebno spreminjati)
params_cl = {
    'cldevices': [clinfo.gpus(vendor)[0],], # vzame prvo grafično z vsebovanim imenom
    'clbuild_options': ['-cl-mad-enable', '-cl-fast-relaxed-math'], # build options (ni potrebno spreminjati)
}

# definicija valovne dolžine svetila
wavelength = 550e-9

# definicija detektorja
px_det = 1e-6  # velikost slikovnega elementa na detektorju
shape_det = (256, 256)  # velikost detektorja za ta primer ustreza cca. 300 x 300 mikronov
x_det = px_det * np.arange(-(shape_det[1]-1)/2, (shape_det[1]-1)/2+1)  # definicija koordinat x
y_det = px_det * np.arange(-(shape_det[0]-1)/2, (shape_det[0]-1)/2+1)  # definicija koordinat y
z_0 = 500e-6  # razdalja med objektom in detektorjem

# definicija objekta
debelina = 20e-6  # debelina objekta
delta_n = 0.05  # razlika lomnih količnikov
px_obj = 0.1e-6  # velikost slikovnega elementa na objektu
shape = (512, 512)  # velikost objekta za ta primer ustreza cca. 50 x 50 mikronov
x_obj = px_obj * np.arange(-(shape[1]-1)/2, (shape[1]-1)/2+1)  # definicija koordinat x
y_obj = px_obj * np.arange(-(shape[0]-1)/2, (shape[0]-1)/2+1)  # definicija koordinat y

# narišemo kroge premera 10 um z neko amplitudo in fazo
r = 5e-6
X, Y = np.meshgrid(x_obj, y_obj)

amplitude = np.ones_like(X)
amplitude[(X**2 + Y**2) <= r**2] = 0.95
amplitude[((X-2*r)**2 + (Y-2*r)**2) <= r**2] = 0.85
amplitude[((X+2*r)**2 + (Y+2*r)**2) <= r**2] = 0.9

phase = np.zeros_like(X)
phase[(X**2 + Y**2) <= r**2] = 2*np.pi/wavelength * delta_n * debelina
phase[((X-2*r)**2 + (Y-2*r)**2) <= r**2] = 2*np.pi/wavelength * delta_n * debelina
phase[((X+2*r)**2 + (Y+2*r)**2) <= r**2] = 2*np.pi/wavelength * delta_n * debelina
field_0 = amplitude * np.exp(1j * phase)  # polje objekta (kot sva definirala na sestanku)

# prikažemo amplitudo objekta
pp.figure()
pp.imshow(np.abs(field_0)**2, cmap='gray')
pp.title('Objekt')
#pp.show()

# definicija polja za Rayleigh-Sommerfeldov algoritem
field = field_0 - sw.PlaneWaveScalar(k=2*np.pi/wavelength).field(0) # Babinet's principle

# definicija propagatorja za Rayleigh-Sommerfeldov algoritem
propagator_rs = rsp.RaySomPix(
    clinfo.gpus(vendor)[0],
    cl_types=rsp.ClTypesDouble,
    cl_build_options=['-cl-mad-enable', '-cl-fast-relaxed-math']
)

propagator_rs.initialize(
    field, 
    pixelsize=(px_obj,px_obj),
    wavelength=wavelength
)

Xi, Yi = np.meshgrid(x_det, y_det)
Zi = np.tile(z_0, Xi.shape)

# izračun polja na detektorju z Rayleigh-Sommerfeldovim algoritmom
t0 = perf_counter()
field_z_rs = propagator_rs.propagate(Xi, Yi, Zi)
field_z_rs += sw.PlaneWaveScalar(k=2*np.pi/wavelength).field(z_0)
print(f'Elapsed time (Rayleigh-Sommerfeld):', perf_counter() - t0)

hologram_z_rs = np.abs(field_z_rs)**2

# prikažemo hologram
pp.figure()
pp.imshow(hologram_z_rs, cmap='gray')
pp.title('Hologram')
#pp.show()

####### ENKRATNA POVRATNA PROPAGACIJA #######
# rekonstrukcija holograma s pomočjo metode kotnega spektra z
# enkratno povratno propagacijo

# definicija propagatorja za metodo kotnega spektra
propagator_asp = asp.Propagator(
    amplitude=np.sqrt(hologram_z_rs),
    phase=np.zeros_like(hologram_z_rs),
    pixelsize=px_det,
    wavelength=wavelength,
    dtype=np.complex128
)

propagator_asp.init_gpu(**params_cl)
propagator_asp.backpropagate(z_0)  # povratna propagacija od detektorja do objekta

field_0_backprop = propagator_asp.field  # rekonstruirano polje objekta z artefaktom dvojne slike

# prikažemo rekonstruirano amplitudo objekta (z dvojno sliko)
# PAZI: px_obj je velikost slikovnega elementa na objektu, zato slika
# objekta ni enaka sliki objekta spodaj, kjer je px_det

pp.figure()
pp.imshow(np.abs(field_0_backprop)**2, cmap='gray')
pp.title('Rekonstrukcija amplitude (dvojna slika)')

####### ITERATIVNA POVRATNA PROPAGACIJA #######
# rekonstrukcija holograma s pomočjo metode kotnega spektra z
# iterativno rekonustrukcijo za elminacijo dvojne slike

# definicija propagatorja za metodo kotnega spektra
propagator_asp = asp.Propagator(
    amplitude=np.sqrt(hologram_z_rs),
    phase=np.zeros_like(hologram_z_rs),
    pixelsize=px_det,
    wavelength=wavelength,
    dtype=np.complex128
)

propagator_asp.init_gpu(**params_cl)

# definicija iteratorja za iterativno rekonstrukcijo
iter = asp.Iterator(
    z=z_0,
    propagator=propagator_asp
)

iter.iterate(iter_num=200) # iterativna rekonstrukcija
field_0_iter = iter.field  # rekonstruirano polje objekta z eliminirano dvojno sliko

# prikažemo rekonstruirano amplitudo objekta (z eliminirano dvojno sliko)
pp.figure()
pp.imshow(np.abs(field_0_iter)**2, cmap='gray')
pp.title('Rekonstrukcija amplitude (brez dvojne slike)')
pp.show()