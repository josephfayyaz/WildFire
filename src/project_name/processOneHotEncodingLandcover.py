import rasterio
import numpy as np
import os
import glob # Per trovare facilmente i file

# --- Configurazione Generale ---
# Assicurati che 'project_root_dir' punti alla radice del tuo progetto
# dove si trova la cartella 'piedmont_new'.
# Ad esempio, se lo script è in una sottocartella, potresti usare:
# current_script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root_dir = os.path.dirname(current_script_dir) # Una cartella sopra lo script
# ROOT_FIRES_FOLDER = os.path.join(project_root_dir, "piedmont_new")

# Per semplicità, assumo che questo script sia eseguito dalla directory genitore di 'piedmont_new'
# O che 'piedmont_new' sia direttamente accessibile dal percorso corrente.
ROOT_FIRES_FOLDER = "piedmont_new"

# --- Parametri Land Cover ---
NO_DATA_VALUE = 0
NUM_ACTUAL_CLASSES = 11 # Le tue classi Land Cover valide vanno da 1 a 11

def process_single_landcover_onehot(input_filepath, output_filepath, no_data_val, num_classes):
    """
    Funzione per leggere un raster Land Cover, applicare il one-hot encoding
    e salvare il risultato come un nuovo raster multi-banda.
    """
    try:
        with rasterio.open(input_filepath) as src:
            landcover_original_data = src.read(1)
            out_profile = src.profile
            
            print(f"  - Letti dati. Dimensione originale: {landcover_original_data.shape}")
            # print(f"  - Valori unici nel raster originale: {np.unique(landcover_original_data)}")
            
            height, width = landcover_original_data.shape

            landcover_one_hot = np.zeros((height, width, num_classes), dtype=np.uint8)
            valid_data_mask = (landcover_original_data > no_data_val)
            
            rows, cols = np.where(valid_data_mask)
            class_indices_0_based = landcover_original_data[valid_data_mask] - 1

            landcover_one_hot[rows, cols, class_indices_0_based] = 1

            print(f"  - One-Hot Encoded completato. Nuova dimensione: {landcover_one_hot.shape}")

            out_profile.update({
                'count': num_classes,
                'dtype': np.uint8,
                'nodata': 0, # I pixel NoData (che erano 0) rimangono tutti 0 nei canali one-hot
                'compress': 'lzw'
            })
            
            with rasterio.open(output_filepath, 'w', **out_profile) as dst:
                dst.write(landcover_one_hot.transpose(2, 0, 1)) # Trasporre per rasterio (C, H, W)
            
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
    print(f"Avvio l'elaborazione one-hot encoding per i file Land Cover in '{ROOT_FIRES_FOLDER}'...")
    
    processed_count = 0
    skipped_count = 0
    
    # Scansiona tutte le sottocartelle in ROOT_FIRES_FOLDER
    for fire_folder_name in os.listdir(ROOT_FIRES_FOLDER):
        fire_folder_path = os.path.join(ROOT_FIRES_FOLDER, fire_folder_name)
        
        # Ci assicuriamo che sia una cartella e che inizi con "fire_"
        if os.path.isdir(fire_folder_path) and fire_folder_name.startswith("fire_"):
            print(f"\nProcessing folder: {fire_folder_name}")
            
            # Costruiamo il nome del file Land Cover basandoci sul nome della cartella
            fire_id = fire_folder_name.replace("fire_", "")
            input_landcover_filename = f"fire_{fire_id}_landcover.tif"
            output_landcover_onehot_filename = f"fire_{fire_id}_landcover_onehot.tif"

            input_full_path = os.path.join(fire_folder_path, input_landcover_filename)
            output_full_path = os.path.join(fire_folder_path, output_landcover_onehot_filename)

            if os.path.exists(input_full_path):
                if os.path.exists(output_full_path):
                    print(f"  - File one-hot '{output_landcover_onehot_filename}' già esistente. Saltato.")
                    skipped_count += 1
                else:
                    print(f"  - Trovato {input_full_path}. Inizio one-hot encoding...")
                    success = process_single_landcover_onehot(
                        input_full_path, 
                        output_full_path, 
                        NO_DATA_VALUE, 
                        NUM_ACTUAL_CLASSES
                    )
                    if success:
                        processed_count += 1
                    else:
                        print(f"  - Fallito il one-hot encoding per {input_landcover_filename}.")
            else:
                print(f"  - File Land Cover '{input_landcover_filename}' non trovato. Saltato.")
                skipped_count += 1
        else:
            print(f"Skipping non-fire folder/file: {fire_folder_name}")
            skipped_count += 1

    print(f"\n--- Processo Completato ---")
    print(f"Cartelle Land Cover processate con successo: {processed_count}")
    print(f"Cartelle/file saltati (già esistenti o non trovati/non-fire): {skipped_count}")
    print("Script terminato.")