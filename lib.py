import torchio as tio
import matplotlib.pyplot as plt
import numpy as np

def plot_nii_image(filename, slice_idx=None, orientation='axial'):
    # Step 1: Load the .nii.gz image using TorchIO
    image = tio.ScalarImage(filename)
    
    # Step 2: Get the data as a NumPy array
    data = image.data.numpy()

    shape = data.shape


    if len(shape) == 4:
        for t in range(shape[3]):
            images = []
            for z in range(shape[0]):
                slice_data = data[z, :, :, t]
                images.append(slice_data)
            fig, axes = plt.subplots(5, 6, figsize=(12, 8))
            for i, ax in enumerate(axes.flat):
                if i < 31:
                    ax.imshow(images[i], cmap='gray')
                    ax.set_title(f'Slice {i} at time {t}')
                    ax.axis('off')
                else:
                    ax.axis('off')  # Hide any extra subplots

            plt.tight_layout()
            plt.show()
        
    
    print(data.shape)
    # Step 3: Determine the slice to plot
    if orientation == 'axial':  # Axial slice (default)
        if slice_idx is None:
            slice_idx = data.shape[2] // 2  # Choose the middle slice if not provided
        slice_data = data[:, :, slice_idx]
    elif orientation == 'sagittal':  # Sagittal slice
        if slice_idx is None:
            slice_idx = data.shape[0] // 2
        slice_data = data[slice_idx, :, :]
    elif orientation == 'coronal':  # Coronal slice
        if slice_idx is None:
            slice_idx = data.shape[1] // 2
        slice_data = data[:, slice_idx, :]
    else:
        raise ValueError("Invalid orientation. Choose from 'axial', 'sagittal', 'coronal'.")
    
    # Step 4: Plot the slice using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data.T, cmap="gray", origin="lower")  # .T to transpose for correct orientation
    plt.title(f'{orientation.capitalize()} Slice {slice_idx}')
    plt.axis('off')
    plt.show()

# Example usage
filename = 'database/testing/patient101/patient101_4d.nii.gz'  # Replace with your .nii.gz file path
plot_nii_image(filename, orientation='axial')
