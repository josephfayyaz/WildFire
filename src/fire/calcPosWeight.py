import os
import rasterio
import numpy as np
from tqdm import tqdm

def calculate_pos_weight(data_root_dir, fire_dirs_list=None):
    """
    Calcola il pos_weight per la loss function in problemi di segmentazione con classi sbilanciate.
    pos_weight = numero_pixel_negativi / numero_pixel_positivi

    Args:
        data_root_dir (str): La directory radice assoluta che contiene le sottocartelle degli incendi.
        fire_dirs_list (list, optional): Una lista di nomi di cartelle (es. ['fire_001', 'fire_002']).
                                          Se None, lo script cercherà tutte le sottocartelle in data_root_dir
                                          che iniziano con 'fire_'.

    Returns:
        float: Il valore di pos_weight. Restituisce np.nan se non ci sono pixel positivi.
    """
    total_positive_pixels = 0
    total_negative_pixels = 0

    if fire_dirs_list is None:
        # Se fire_dirs_list non è fornito, cerchiamo tutte le sottocartelle in data_root_dir
        fire_dirs = sorted([d for d in os.listdir(data_root_dir)
                            if os.path.isdir(os.path.join(data_root_dir, d)) and d.startswith('fire_')])
    else:
        fire_dirs = fire_dirs_list

    print(f"Processing {len(fire_dirs)} fire directories...")
    print(f"Base data directory (DATA_ROOT): {data_root_dir}") # DEBUG

    for fire_dir_name in tqdm(fire_dirs, desc="Calculating pos_weight"):
        fire_dir_path = os.path.join(data_root_dir, fire_dir_name)
        
        # DEBUG: Stampa il percorso completo della cartella dell'incendio
        # print(f"DEBUG: fire_dir_path = {fire_dir_path}") 

        # Estrai il prefisso numerico dalla cartella (es. '0001' da 'fire_0001')
        try:
            fire_id = fire_dir_name.split('_')[1] # Ottiene 'XXXX' da 'fire_XXXX'
            if not fire_id.isdigit(): # Assicurati che sia effettivamente un ID numerico
                 raise ValueError("fire_id is not purely numeric.")
        except (IndexError, ValueError):
            print(f"Warning: Directory name '{fire_dir_name}' does not match expected format 'fire_XXXX'. Skipping.")
            continue

        # Costruisci il nome del file GTSentinel basandoti sul nome della cartella
        gt_filename = f"fire_{fire_id}_GTSentinel.tif"
        gt_path = os.path.join(fire_dir_path, gt_filename)

        # DEBUG: Stampa il percorso completo del file GT che sta cercando
        # print(f"DEBUG: Looking for GT at: {gt_path}") 

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file '{gt_filename}' not found for {fire_dir_name} at {gt_path}. Skipping.")
            continue

        try:
            with rasterio.open(gt_path) as src:
                mask = src.read(1) # Legge il primo canale
                
                # Assicurati che la maschera sia binaria (0 o 1)
                mask = (mask > 0).astype(np.uint8) 

                positive_pixels = np.sum(mask == 1)
                negative_pixels = np.sum(mask == 0)

                total_positive_pixels += positive_pixels
                total_negative_pixels += negative_pixels

        except Exception as e:
            print(f"Error processing {gt_path}: {e}. Skipping.")
            continue

    if total_positive_pixels == 0:
        print("Warning: No positive (fire) pixels found in the dataset. pos_weight cannot be calculated.")
        return np.nan # Not a number, per indicare l'impossibilità di calcolare

    pos_weight = total_negative_pixels / total_positive_pixels
    print(f"\n--- Calculation Complete ---")
    print(f"Total positive (fire) pixels: {total_positive_pixels}")
    print(f"Total negative (non-fire) pixels: {total_negative_pixels}")
    print(f"Calculated pos_weight: {pos_weight:.4f}")

    return pos_weight

if __name__ == "__main__":
    # --- INIZIO MODIFICA CRUCIALE PER LA TUA STRUTTURA DI CARTELLE ---
    # Ottieni il percorso assoluto della directory corrente dello script (es. /path/to/tuo_progetto/src/fire/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Risali di due livelli per arrivare a 'tuo_progetto/'
    # current_script_dir -> /path/to/tuo_progetto/src/fire/
    # os.path.dirname(current_script_dir) -> /path/to/tuo_progetto/src/
    # os.path.dirname(os.path.dirname(current_script_dir)) -> /path/to/tuo_progetto/
    project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
    
    # Ora costruisci il percorso assoluto a 'piedmont_new'
    DATA_ROOT = os.path.join(project_root_dir, 'piedmont_new') 
    # --- FINE MODIFICA CRUCIALE ---

    # --- Configurazione delle cartelle da analizzare ---
    # Se hai già generato una lista `all_filtered_dirs` o `train_dirs` nel tuo `main.py`,
    # potresti volerla passare qui dopo averla resa persistente (es. con JSON).
    fire_dirs_to_use_for_pos_weight = None 

    pos_weight_value = calculate_pos_weight(DATA_ROOT, fire_dirs_to_use_for_pos_weight)

    if not np.isnan(pos_weight_value):
        print(f"\nRecommended pos_weight for your loss function: {pos_weight_value:.4f}")
        print("You can use this value as 'pos_weight' in nn.BCEWithLogitsLoss or HybridLoss.")
    else:
        print("Could not calculate pos_weight due to no positive pixels or errors (e.g., no fire pixels found).")