
import os
import csv
import rasterio
import numpy as np

# Percorso della cartella principale
base_path = 'piedmont_new'
soglia_vuoti = 0.2  # 40%

# Lista dei risultati
risultati = []

print(f'Inizio analisi dei file TIFF in "{base_path}"...')

for root, dirs, files in os.walk(base_path):
    for file in files:
        # Modifica: Filtra per tutti i file TIFF generati dai tuoi script
        if ('landsat'in file or 'sentinel' in file) and file.endswith('.tif') and "GT" not in file:
            file_path = os.path.join(root, file)
            
            try:
                with rasterio.open(file_path) as src:
                    # Leggi il valore NoData dal profilo del raster
                    nodata_value = src.nodata
                    
                    if nodata_value is None:
                        # Se il valore NoData non è specificato nel file, assumiamo che non ci siano dati mancanti
                        # Oppure puoi impostare un valore di default, es. 0
                        nodata_value = 0 # Valore di default
                        # Emettiamo un avvertimento per maggiore chiarezza
                        # print(f"⚠️ Attenzione: il file '{file}' non ha un valore NoData definito. Assumo che sia {nodata_value}.")
                    
                    # Leggi tutte le bande
                    data_cube = src.read()
                    
                    # Calcola il numero totale di pixel
                    total_pixels = data_cube.size / data_cube.shape[0] # O src.width * src.height
                    
                    # Calcola il numero di pixel "vuoti" in tutte le bande.
                    # Un pixel è vuoto se è NoData in *qualsiasi* banda.
                    nodata_mask = np.zeros(src.shape, dtype=bool)
                    for band_idx in range(src.count):
                        band_data = data_cube[band_idx]
                        nodata_mask = nodata_mask | (band_data == nodata_value)

                    # Calcola la percentuale di pixel NoData
                    pixels_nodata_count = np.sum(nodata_mask)
                    percentuale_nodata = (pixels_nodata_count / total_pixels) * 100
                    
                    supera_soglia = percentuale_nodata / 100 > soglia_vuoti
                    
                    risultati.append([
                        file_path,
                        f'{percentuale_nodata:.2f}%',
                        'YES' if supera_soglia else 'NO'
                    ])
                    
            except Exception as e:
                print(f'Errore durante l\'analisi di {file_path}: {e}')

# Salva su CSV
output_dir = 'src/project_name/eliminateNoData'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'pixel_nodata_pre.csv')

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'percentuale_nodata', 'supera_soglia'])
    writer.writerows(risultati)

print(f'\nAnalisi completata. Risultati salvati in "{output_csv}"')

