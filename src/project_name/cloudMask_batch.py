import os
import re
import numpy as np
import rasterio
from cloudsen12_models.cloudsen12 import load_model_by_name
from PIL import Image

# === PARAMETRI ===
root_dir = "piedmont_new"
model_name = "cloudsen12l2a"
expected_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
band_map = {
    'B01': 'B1', 'B02': 'B2', 'B03': 'B3', 'B04': 'B4',
    'B05': 'B5', 'B06': 'B6', 'B07': 'B7', 'B08': 'B8',
    'B8A': 'B8A', 'B09': 'B9', 'B11': 'B11', 'B12': 'B12'
}

# === Caricamento modello una sola volta ===
print(f"Caricamento del modello: {model_name}...")
model = load_model_by_name(model_name)
print("Modello caricato.\n")

# === Scansione ricorsiva della cartella ===
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if not filename.endswith(".tif"):
            continue

        tif_path = os.path.join(dirpath, filename)

        # Esclude i non-sentinel e GTSentinel
        if ("GTSentinel" in filename or "sentinel" not in filename ):
            continue

        print(f"Elaborazione file: {tif_path}")

        try:
            # === Lettura e ordinamento bande ===
            with rasterio.open(tif_path) as src:
                band_names = src.descriptions
                band_indices = {band_map[b]: i for i, b in enumerate(band_names) if b in band_map}

                if not all(b in band_indices for b in expected_bands):
                    missing = [b for b in expected_bands if b not in band_indices]
                    print(f"❌ Bande mancanti in {tif_path}: {missing}")
                    continue

                ordered_indices = [band_indices[b] for b in expected_bands]
                bands = [src.read(idx + 1).astype(np.float32) for idx in ordered_indices]
                arr = np.stack(bands) / 10_000

                meta = src.meta.copy()
                transform = src.transform
                crs = src.crs

            # === Predizione ===
            print(" → Predizione della maschera nuvole...")
            mask = model.predict(arr)
            mask_tif = mask.astype(rasterio.uint8)

            # === Salvataggio output ===
            out_tif_path = tif_path.replace(".tif", "_CM.tif")
            meta.update({
                'count': 1,
                'dtype': rasterio.uint8,
                'nodata': 255
            })

            with rasterio.open(out_tif_path, 'w', **meta) as dst:
                dst.write(mask_tif, 1)

            print(f"✔ Maschera nuvole salvata in: {out_tif_path}\n")

        except Exception as e:
            print(f"❌ Errore con {tif_path}: {e}\n")
