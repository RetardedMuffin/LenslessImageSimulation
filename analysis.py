import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where the .npz files are stored
npz_dir = 'hologram'

# List all .npz files in the directory
npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

# Loop through each .npz file
for npz_file in npz_files:
    # Load the .npz file
    npz_path = os.path.join(npz_dir, npz_file)
    data = np.load(npz_path)
    
    # Assuming the file contains arrays, we can load them
    # For example, let's assume 'hologram' is the key in the .npz file
    hologram = data['hologram_z_rs']  # Replace 'hologram' with the actual key name

    # Display the hologram image
    plt.figure()
    plt.imshow(hologram, cmap='gray')
    plt.title(f'Hologram: {npz_file}')
    plt.colorbar()

    # If you want to process the hologram data further, you can do so here
    # For example, you might want to save it as a .png file:
    output_png_path = os.path.join(npz_dir, f'{os.path.splitext(npz_file)[0]}.png')
    plt.imsave(output_png_path, hologram, cmap='gray')
    # Close the figure to free up memory
    plt.close()
    