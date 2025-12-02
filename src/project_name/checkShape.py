import rasterio
import numpy as np
import matplotlib.pyplot as plt # For visualization

# Path to your sample GT file
gt_path = "piedmont_new/fire_4938/fire_4938_landcover.tif"

with rasterio.open(gt_path) as src:
    print("--- Rasterio Metadata ---")
    print("Transform:\n", src.transform)
    print("Descriptions:", src.descriptions)
    print(f"Shapes (height, width): {src.height}, {src.width}")
    print("CRS:", src.crs)
    print(f"Number of bands: {src.count}") # How many bands does this raster have?
    print(f"Data type (rasterio): {src.dtypes}") # Data type as identified by rasterio

    # --- Read the pixel data ---
    # Read the first band (most masks are single-band)
    # If your mask has multiple bands and you only need one, specify the band number (1-indexed)
    # If it's a multi-class mask where each band is a class, you might read all bands.
    # Assuming it's a single-band binary mask:
    mask_data = src.read(1) 

    print("\n--- Pixel Data Analysis ---")
    print(f"NumPy array shape: {mask_data.shape}")
    print(f"NumPy array data type: {mask_data.dtype}")

    # Get unique pixel values
    unique_values = np.unique(mask_data)
    print(f"Unique pixel values: {unique_values}")

    # Get counts of unique pixel values (useful for binary masks)
    # Note: bincount works best for non-negative integers.
    if np.issubdtype(mask_data.dtype, np.integer):
        # Flatten the array to use bincount
        flattened_mask = mask_data.flatten()
        
        # Determine the maximum value to ensure bincount array is large enough
        max_val = np.max(flattened_mask)
        
        # Create bins up to max_val + 1
        counts = np.bincount(flattened_mask, minlength=max_val + 1)
        
        print("Counts of unique pixel values:")
        #for val in unique_values:
        #    print(f"  Value {val}: {counts[val]} pixels")
    else:
        print("Cannot use np.bincount directly on float data types. Unique counts are:")
        unique_vals, counts = np.unique(mask_data, return_counts=True)
        #for val, count in zip(unique_vals, counts):
           # print(f"  Value {val}: {count} pixels")


    # Get min and max pixel values
    min_val = np.min(mask_data)
    max_val = np.max(mask_data)
    print(f"Minimum pixel value: {min_val}")
    print(f"Maximum pixel value: {max_val}")
'''
    # --- Visualize the mask ---
    print("\n--- Mask Visualization ---")
    plt.figure(figsize=(6, 6))
    # Use 'gray' colormap for binary or single-channel masks
    # Use interpolation='nearest' to see sharp pixels, useful for masks
    plt.imshow(mask_data, cmap='gray', interpolation='nearest') 
    plt.title(f"GT Mask: {os.path.basename(gt_path)}")
    plt.colorbar(label="Pixel Value")
    plt.show()
'''