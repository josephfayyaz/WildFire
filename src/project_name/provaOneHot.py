import rasterio
import numpy as np
import os

# --- Configurazione ---
# Assicurati che il percorso sia corretto rispetto a dove esegui lo script.
# Se lo script è nella stessa directory di 'piedmont_new':
input_file_path = os.path.join("piedmont_new", "fire_6500", "fire_6500_streets.tif")
output_file_path = os.path.join("piedmont_new", "fire_6500", "fire_6500_streets_onehot.tif")

# --- Parametri Streets ---
NO_DATA_VALUE_STREETS = 0
NUM_ACTUAL_CLASSES_STREETS = 5 # Le tue classi Streets valide vanno da 1 a 5

print(f"Tentativo di leggere il file Streets: {input_file_path}")

try:
    with rasterio.open(input_file_path) as src:
        # Leggi la singola banda delle Streets
        streets_original_data = src.read(1)
        
        # Copia il profilo del dataset originale per il nuovo file di output
        out_profile = src.profile
        
        print(f"File letto con successo. Dimensione originale: {streets_original_data.shape}")
        print(f"Valori unici nel raster Streets originale: {np.unique(streets_original_data)}")
        print(f"NoData originale (se definito in TIFF): {src.nodata}")
        
        height, width = streets_original_data.shape

        # --- ONE-HOT ENCODING PER STREETS ---
        # Inizializza l'array one-hot con tutti zeri.
        streets_one_hot = np.zeros((height, width, NUM_ACTUAL_CLASSES_STREETS), dtype=np.uint8)

        # Crea una maschera booleana per identificare i pixel con dati validi (non NoData)
        valid_data_mask = (streets_original_data > NO_DATA_VALUE_STREETS)

        # Ottieni gli indici di riga e colonna per i pixel con dati validi
        rows, cols = np.where(valid_data_mask)

        # Ottieni i valori delle classi per i pixel validi
        # E convertili in indici 0-based: classe 1 -> indice 0, classe 5 -> indice 4
        class_indices_0_based = streets_original_data[valid_data_mask] - 1

        # Usa l'indicizzazione avanzata per impostare a 1 solo i canali corrispondenti
        streets_one_hot[rows, cols, class_indices_0_based] = 1

        print(f"Streets One-Hot Encoded. Nuova dimensione: {streets_one_hot.shape}")
        print(f"Numero di canali (bande) one-hot: {streets_one_hot.shape[2]}")

        # --- Salvataggio del file TIFF multi-banda ---
        
        # Aggiorna il profilo per il nuovo file
        out_profile.update({
            'count': NUM_ACTUAL_CLASSES_STREETS, # Numero di bande = numero di classi
            'dtype': np.uint8,                   # Il one-hot encoding usa 0 e 1, quindi uint8 è sufficiente
            'nodata': 0,                         # I pixel NoData rimangono 0 su tutti i canali
            'compress': 'lzw'                    # Compressione
        })
        
        print(f"Salvataggio del file one-hot encoded in: {output_file_path}")
        with rasterio.open(output_file_path, 'w', **out_profile) as dst:
            # rasterio si aspetta le bande nell'ordine (banda, altezza, larghezza)
            # Quindi dobbiamo trasporre l'array one-hot da (H, W, C) a (C, H, W)
            dst.write(streets_one_hot.transpose(2, 0, 1))
        
        print("Salvataggio completato. Puoi ispezionare il file con un software GIS (es. QGIS).")

except rasterio.errors.RasterioIOError as e:
    print(f"ERRORE: Impossibile leggere il file TIFF. Controlla il percorso: {input_file_path}")
    print(f"Errore dettagliato: {e}")
except Exception as e:
    print(f"Si è verificato un errore inatteso: {e}")