import rasterio
import numpy as np
import os
import glob

# --- Configurazione Generale ---
# Assicurati che 'ROOT_FIRES_FOLDER' punti alla radice del tuo progetto
# dove si trova la cartella 'piedmont_new'.
ROOT_FIRES_FOLDER = "piedmont_new"

# --- Parametri Streets ---
NO_DATA_VALUE_STREETS = 0
NUM_ACTUAL_CLASSES_STREETS = 5 # Le tue classi Streets valide vanno da 1 a 5

def process_single_raster_onehot(input_filepath, output_filepath, no_data_val, num_classes):
    """
    Funzione per leggere un raster, applicare il one-hot encoding
    e salvare il risultato come un nuovo raster multi-banda.
    """
    try:
        with rasterio.open(input_filepath) as src:
            original_data = src.read(1) # Leggi la singola banda
            out_profile = src.profile
            
            print(f"  - Letti dati. Dimensione originale: {original_data.shape}")
            # print(f"  - Valori unici nel raster originale: {np.unique(original_data)}")
            
            height, width = original_data.shape

            one_hot_encoded = np.zeros((height, width, num_classes), dtype=np.uint8)
            valid_data_mask = (original_data > no_data_val)
            
            rows, cols = np.where(valid_data_mask)
            class_indices_0_based = original_data[valid_data_mask] - 1

            one_hot_encoded[rows, cols, class_indices_0_based] = 1

            print(f"  - One-Hot Encoded completato. Nuova dimensione: {one_hot_encoded.shape}")

            out_profile.update({
                'count': num_classes,
                'dtype': np.uint8,
                'nodata': 0, 
                'compress': 'lzw'
            })
            
            with rasterio.open(output_filepath, 'w', **out_profile) as dst:
                dst.write(one_hot_encoded.transpose(2, 0, 1)) # Trasporre per rasterio (C, H, W)
            
            print(f"  - Salvato {output_filepath}")
            return True

    except rasterio.errors.RasterioIOError as e:
        print(f"  ERRORE: Impossibile leggere o scrivere il file TIFF: {e}")
        return False
    except Exception as e:
        print(f"  ERRORE INATTESO durante l'elaborazione di {input_filepath}: {e}")
        return False

# --- Processo Principale ---
if not os.path.isdir(ROOT_FIRES_FOLDER):
    print(f"ERRORE: La cartella '{ROOT_FIRES_FOLDER}' non esiste. Controlla il percorso.")
else:
    print(f"Avvio l'elaborazione one-hot encoding per i file Streets in '{ROOT_FIRES_FOLDER}'...")
    
    processed_count = 0
    skipped_count = 0
    
    # Scansiona tutte le sottocartelle in ROOT_FIRES_FOLDER
    for fire_folder_name in os.listdir(ROOT_FIRES_FOLDER):
        fire_folder_path = os.path.join(ROOT_FIRES_FOLDER, fire_folder_name)
        
        # Ci assicuriamo che sia una cartella e che inizi con "fire_"
        if os.path.isdir(fire_folder_path) and fire_folder_name.startswith("fire_"):
            print(f"\nProcessing folder: {fire_folder_name}")
            
            # Costruiamo il nome del file Streets basandoci sul nome della cartella
            fire_id = fire_folder_name.replace("fire_", "")
            input_streets_filename = f"fire_{fire_id}_streets.tif"
            output_streets_onehot_filename = f"fire_{fire_id}_streets_onehot.tif"

            input_full_path = os.path.join(fire_folder_path, input_streets_filename)
            output_full_path = os.path.join(fire_folder_path, output_streets_onehot_filename)

            if os.path.exists(input_full_path):
                if os.path.exists(output_full_path):
                    print(f"  - File one-hot '{output_streets_onehot_filename}' già esistente. Saltato.")
                    skipped_count += 1
                else:
                    print(f"  - Trovato {input_full_path}. Inizio one-hot encoding...")
                    success = process_single_raster_onehot(
                        input_full_path, 
                        output_full_path, 
                        NO_DATA_VALUE_STREETS, 
                        NUM_ACTUAL_CLASSES_STREETS
                    )
                    if success:
                        processed_count += 1
                    else:
                        print(f"  - Fallito il one-hot encoding per {input_streets_filename}.")
            else:
                print(f"  - File Streets '{input_streets_filename}' non trovato. Saltato.")
                skipped_count += 1
        else:
            print(f"Skipping non-fire folder/file: {fire_folder_name}")
            skipped_count += 1

    print(f"\n--- Processo Completato ---")
    print(f"Cartelle Streets processate con successo: {processed_count}")
    print(f"Cartelle/file saltati (già esistenti o non trovati/non-fire): {skipped_count}")
    print("Script terminato.")