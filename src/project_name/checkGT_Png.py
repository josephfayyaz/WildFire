import rasterio
import numpy as np
from PIL import Image

def save_gt_as_png(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        mask = src.read(1)  # Read single band

    # Create an RGB image (3D array)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply colors: 0 = black, 1 = red
    rgb[mask == 1] = [255, 0, 0]  # Red
    # Background stays black ([0,0,0])

    # Convert to image and save
    img = Image.fromarray(rgb)
    img.save(png_path)
    print(f"Saved PNG: {png_path}")

save_gt_as_png(
    tif_path='piedmont/fire_6207/fire_6207_GTSentinel.tif',
    png_path='piedmont/fire_6207/fire_6207_GTSentinel.png'
)