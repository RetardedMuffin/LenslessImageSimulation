from time import perf_counter

import numpy as np
import matplotlib.pyplot as pp

import clinfo
import raysompix as rsp
import angularspectrum as asp
import scalarwave as sw

from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None 

# definicija za grafično kartico
vendor = 'Nvidia' # del imena grafične kartice
datatype = np.complex128 # računska natančnost (ni potrebno spreminjati)
params_cl = {
    'cldevices': [clinfo.gpus(vendor)[0],], # vzame prvo grafično z vsebovanim imenom
    'clbuild_options': ['-cl-mad-enable', '-cl-fast-relaxed-math'], # build options (ni potrebno spreminjati)
}

# ustvarimo mape za shranjevanje slik
output_dir1 = 'hologram'
#output_dir2 = 'backpropagation'
#output_dir3 = 'iteration'

os.makedirs(output_dir1, exist_ok=True)
# os.makedirs(output_dir2, exist_ok=True)
# os.makedirs(output_dir3, exist_ok=True)

# nalaganje tarče
img = Image.open('raster_usaf.png')
amplitude = np.array(img) / 255.0
phase = np.zeros_like(amplitude)

field_0 = amplitude * np.exp(1j * phase)  # polje objekta (kot sva definirala na sestanku)

# definicija parametrov
wavelengths = [400e-9, 450e-9, 500e-9, 550e-9, 600e-9, 650e-9, 700e-9, 750e-9] # valovna dolžina svetila
px_dets = [0.5e-6]#, 1e-6, 1.5e-6, 2.2e-6, 3e-6, 3.45e-6] # velikost slikovnega elementa na detektorju
z_0s = [100e-6, 0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3] # razdalja med objektom in detektorjem

# definicija objekta
debelina = 20e-6  # debelina objekta
delta_n = 0.05  # razlika lomnih količnikov
px_obj = 0.1e-6  # velikost slikovnega elementa na objektu
shape = (512, 512)  # velikost objekta za ta primer ustreza cca. 50 x 50 mikronov
x_obj = px_obj * np.arange(-(shape[1]-1)/2, (shape[1]-1)/2+1)  # definicija koordinat x
y_obj = px_obj * np.arange(-(shape[0]-1)/2, (shape[0]-1)/2+1)  # definicija koordinat y

for wavelength in wavelengths:
    for px_det in px_dets:
        # definicija detektorja
        shape_det = (512, 512)  # velikost detektorja za ta primer ustreza cca. 300 x 300 mikronov
        x_det = px_det * np.arange(-(shape_det[1]-1)/2, (shape_det[1]-1)/2+1)  # definicija koordinat x
        y_det = px_det * np.arange(-(shape_det[0]-1)/2, (shape_det[0]-1)/2+1)  # definicija koordinat y
        
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

        for z_0 in z_0s:
            Zi = np.tile(z_0, Xi.shape)

            # izračun polja na detektorju z Rayleigh-Sommerfeldovim algoritmom
            t0 = perf_counter()
            field_z_rs = propagator_rs.propagate(Xi, Yi, Zi)
            field_z_rs += sw.PlaneWaveScalar(k=2*np.pi/wavelength).field(z_0)
            print(f'Elapsed time (Rayleigh-Sommerfeld):', perf_counter() - t0)

            hologram_z_rs = np.abs(field_z_rs)**2

            np.savez(
                os.path.join(output_dir1, f'hologram_lambda_{int(wavelength*1e9)}nm_pxdet_{int(px_det*1e6)}um_z0_{int(z_0*1e6)}um.npz'),
                hologram_z_rs=hologram_z_rs, field_z_rs=field_z_rs
            )

            print('Completion:', f'{(wavelengths.index(wavelength)*len(px_dets)*len(z_0s) + px_dets.index(px_det)*len(z_0s) + z_0s.index(z_0) + 1) / (len(wavelengths)*len(px_dets)*len(z_0s))*100}%', end='\r')

            # ####### ENKRATNA POVRATNA PROPAGACIJA #######
            # # rekonstrukcija holograma s pomočjo metode kotnega spektra z
            # # enkratno povratno propagacijo

            # # definicija propagatorja za metodo kotnega spektra
            # propagator_asp = asp.Propagator(
            #     amplitude=np.sqrt(hologram_z_rs),
            #     phase=np.zeros_like(hologram_z_rs),
            #     pixelsize=px_det,
            #     wavelength=wavelength,
            #     dtype=np.complex128
            # )


            # propagator_asp.init_gpu(**params_cl)
            # propagator_asp.backpropagate(z_0)  # povratna propagacija od detektorja do objekta

            # field_0_backprop = propagator_asp.field  # rekonstruirano polje objekta z artefaktom dvojne slike

            
            # # definicija propagatorja za metodo kotnega spektra
            # propagator_asp = asp.Propagator(
            #     amplitude=np.sqrt(hologram_z_rs),
            #     phase=np.zeros_like(hologram_z_rs),
            #     pixelsize=px_det,
            #     wavelength=wavelength,
            #     dtype=np.complex128
            # )

            # propagator_asp.init_gpu(**params_cl)

            # # definicija iteratorja za iterativno rekonstrukcijo
            # iter = asp.Iterator(
            #     z=z_0,
            #     propagator=propagator_asp
            # )

            # iter.iterate(iter_num=200) # iterativna rekonstrukcija
            # field_0_iter = iter.field  # rekonstruirano polje objekta z eliminirano dvojno sliko


            # pp.figure()
            # pp.imshow(hologram_z_rs, cmap="gray")
            # pp.set_title('Hologram objekta')
            # pp.show()
            # pp.savefig(os.path.join(output_dir1, f'hologram_lambda_{int(wavelength*1e9)}nm_pxdet_{int(px_det*1e6)}um_z0_{int(z_0*1e6)}um.png'))

            # pp.figure()
            # pp.imshow(np.abs(field_0_backprop)**2, cmap='gray')
            # pp.set_title('Rekonstrukcija amplitude (dvojna slika)')
            # pp.savefig(os.path.join(output_dir2, f'backpropagate_lambda_{int(wavelength*1e9)}nm_pxdet_{int(px_det*1e6)}um_z0_{int(z_0*1e6)}um.png'))

            # pp.figure()
            # pp.imshow(np.abs(field_0_iter)**2, cmap='gray')
            # pp.set_title('Rekonstrukcija amplitude (brez dvojne slike)')
            # pp.savefig(os.path.join(output_dir3, f'iterReconst_lambda_{int(wavelength*1e9)}nm_pxdet_{int(px_det*1e6)}um_z0_{int(z_0*1e6)}um.png'))