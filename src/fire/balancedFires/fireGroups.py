import os
import rasterio
import numpy as np

def create_balanced_id_lists(root_folder: str, small_threshold: int, large_threshold: int):
    """
    Scorre tutte le cartelle degli incendi, calcola la dimensione di ogni incendio
    e salva gli ID in liste separate (piccoli, medi, grandi) SOLO se è presente
    almeno un file Sentinel-2.
    
    Args:
        root_folder (str): Il percorso della cartella radice (es. 'piedmont_new').
        small_threshold (int): La soglia per definire un incendio "piccolo".
        large_threshold (int): La soglia per definire un incendio "grande".
    """
    if not os.path.isdir(root_folder):
        print(f"ERRORE: La cartella '{root_folder}' non esiste. Verifica il percorso.")
        return

    small_fire_ids = []
    medium_fire_ids = []
    large_fire_ids = []
    
    print(f"Inizio la creazione delle liste di ID bilanciate in '{root_folder}'...")

    for dirpath, dirnames, filenames in os.walk(root_folder):
        if os.path.basename(dirpath).startswith("fire_"):
            fire_id = os.path.basename(dirpath).replace("fire_", "")
            
            gt_file_path = os.path.join(dirpath, f"fire_{fire_id}_GTSentinel.tif")
            
            # --- MODIFICA CHIAVE ---
            # Utilizza la tua logica per cercare un file Sentinel generico
            has_sentinel = any("pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f for f in filenames)
            
            if os.path.exists(gt_file_path) and has_sentinel:
            # --- FINE MODIFICA ---
                try:
                    with rasterio.open(gt_file_path) as src:
                        gt_data = src.read(1)
                        burned_pixel_count = np.sum(gt_data == 1)
                        
                        if burned_pixel_count <= small_threshold:
                            small_fire_ids.append(fire_id)
                        elif burned_pixel_count > large_threshold:
                            large_fire_ids.append(fire_id)
                        else:
                            medium_fire_ids.append(fire_id)
                except Exception as e:
                    print(f"❌ Errore durante l'apertura del file {gt_file_path}: {e}")
            else:
                print(f"⚠️ ATTENZIONE: File GT o Sentinel non trovato per l'incendio ID {fire_id}. Directory ignorata.")

    # Stampa i conteggi
    print("\n--- Riepilogo dei Gruppi ---")
    print(f"Numero di incendi Piccoli: {len(small_fire_ids)}")
    print(f"Numero di incendi Medi: {len(medium_fire_ids)}")
    print(f"Numero di incendi Grandi: {len(large_fire_ids)}")

    # Salva gli ID in file di testo per un facile riutilizzo
    with open("src/fire/balancedFires/small_fire_ids.txt", "w") as f:
        f.write('\n'.join(small_fire_ids))
    
    with open("src/fire/balancedFires/medium_fire_ids.txt", "w") as f:
        f.write('\n'.join(medium_fire_ids))

    with open("src/fire/balancedFires/large_fire_ids.txt", "w") as f:
        f.write('\n'.join(large_fire_ids))
        
    print("\n🎉 Liste di ID salvate in 'small_fire_ids.txt', 'medium_fire_ids.txt', 'large_fire_ids.txt'.")

if __name__ == "__main__":
    root_directory = "piedmont_new"
    small_threshold = 2000
    large_threshold = 10000
    
    print(f"Soglia per incendi Piccoli: <= {small_threshold} pixel")
    print(f"Soglia per incendi Grandi: > {large_threshold} pixel")
    
    create_balanced_id_lists(root_directory, small_threshold, large_threshold)