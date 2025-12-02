import os
import rasterio
import numpy as np
import cv2
from modelWithLandsat.utils import resample_image, read_image_and_metadata
from rasterio.warp import Resampling

# --- Configurazione ---
FIRE_DIRECTORY = 'piedmont_new/fire_6500' 
# Sostituisci con i nomi dei tuoi file reali
SENTINEL_IMAGE_NAME = 'fire_6500_2021-08-14_pre_sentinel_1.tif'
LANDSAT_IMAGE_NAME = 'fire_6500_dem.tif'
OUTPUT_DIR = 'resampling_debug_output'

# Crea la cartella di output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Esegui il Resampling ---

# Passo 1: Carica i metadati di Landsat (il nostro riferimento a 30m)
landsat_path = os.path.join(FIRE_DIRECTORY, LANDSAT_IMAGE_NAME)
if not os.path.exists(landsat_path):
    print(f"Errore: File Landsat non trovato a {landsat_path}. Aggiorna il percorso.")
    exit()

# ATTENZIONE: MODIFICA QUI per spacchettare il valore extra
_, landsat_transform, landsat_crs, landsat_shape, landsat_res, landsat_band_descriptions = read_image_and_metadata(landsat_path)
print(f"Landsat metadata: shape={landsat_shape}, res={landsat_res}m")
print(f"Landsat band descriptions: {landsat_band_descriptions}")

# Passo 2: Carica i metadati di Sentinel (il nostro riferimento a 10m)
sentinel_path = os.path.join(FIRE_DIRECTORY, SENTINEL_IMAGE_NAME)
if not os.path.exists(sentinel_path):
    print(f"Errore: File Sentinel non trovato a {sentinel_path}. Aggiorna il percorso.")
    exit()

# ATTENZIONE: MODIFICA ANCHE QUI per spacchettare il valore extra
_, sentinel_transform, sentinel_crs, sentinel_shape, sentinel_res, sentinel_band_descriptions = read_image_and_metadata(sentinel_path)
print(f"Sentinel metadata: shape={sentinel_shape}, res={sentinel_res}m")
print(f"Sentinel band descriptions: {sentinel_band_descriptions}")

'''
# Esempio A: Downsampling di Sentinel a 30m (il tuo caso d'uso principale)
print("\n--- Eseguo il DOWNsampling di Sentinel a 30m ---")
output_sentinel_30m_path = os.path.join(OUTPUT_DIR, 'sentinel_resampled_to_30m.tif')
try:
    resample_image(
        input_tif_path=sentinel_path,
        output_tif_path=output_sentinel_30m_path,
        target_transform=landsat_transform,
        target_crs=landsat_crs,
        target_shape=landsat_shape # Passa (Height, Width)
    )
except Exception as e:
    print(f"Errore nel downsampling di Sentinel: {e}")
'''
# Esempio B: Upsampling di Landsat a 10m (il caso d'uso alternativo)
print("\n--- Eseguo l'UPsampling di Landsat a 10m ---")
output_landsat_10m_path = os.path.join(OUTPUT_DIR, 'fire_6500_dem_10m.tif')
try:
    resample_image(
        input_tif_path=landsat_path,
        output_tif_path=output_landsat_10m_path,
        target_transform=sentinel_transform,
        target_crs=sentinel_crs,
        target_shape=sentinel_shape # Passa (Height, Width)
    )
except Exception as e:
    print(f"Errore nell'upsampling di Landsat: {e}")

print("\nScript di debug completato. Controlla la cartella 'resampling_debug_output'.")