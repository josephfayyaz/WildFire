import os
import csv
import rasterio
import numpy as np

# Percorso della cartella principale
base_path = 'piedmont'
soglia_vuoti = 0.4  # 40%

# Lista dei risultati
risultati = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if 'pre_landsat' in file and file.endswith('.tif'):
            file_path = os.path.join(root, file)
            try:
                with rasterio.open(file_path) as src:
                    band = src.read(1)  # Legge la prima banda  # Legge la maschera della prima banda
                    pixel_nodata= np.sum(band == 0) / band.size * 100

                    supera_soglia = pixel_nodata/100 > soglia_vuoti
                    risultati.append([
                        file_path,
                        pixel_nodata,
                        'YES' if supera_soglia else 'NO'
                    ])
            except Exception as e:
                print(f'Errore con {file_path}: {e}')
# Salva su CSV
output_csv = 'src/fire/eliminateNoData/pixel_nodata_landsat_pre.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'pixel_nodata', 'supera_soglia_40%'])
    writer.writerows(risultati)

print(f'\nAnalisi completata. Risultati salvati in "{output_csv}"')