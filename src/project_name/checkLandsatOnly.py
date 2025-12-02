import os

def count_landsat_only_fires(root_folder: str):
    """
    Conta il numero di cartelle "fire_XXX" che contengono dati Landsat
    ma non dati Sentinel.
    """
    print(f"Analisi delle cartelle in: {root_folder}")
    
    landsat_only_count = 0
    total_fires = 0

    if not os.path.exists(root_folder):
        print(f"ERRORE: La cartella radice '{root_folder}' non esiste.")
        return 0

    for item in os.listdir(root_folder):
        fire_folder_path = os.path.join(root_folder, item)
        
        if os.path.isdir(fire_folder_path) and item.startswith("fire_"):
            total_fires += 1
            fire_id = item.replace("fire_", "")

            has_sentinel = False
            has_landsat = False

            # Controlla per i file Sentinel
            # Adatta il pattern se i tuoi nomi file Sentinel sono diversi
            for fname in os.listdir(fire_folder_path):
                if "pre_sentinel_" in fname and fname.endswith(".tif"):
                    has_sentinel = True
                    break
            
            # Controlla per i file Landsat
            # Adatta il pattern se i tuoi nomi file Landsat sono diversi
            for fname in os.listdir(fire_folder_path):
                if "pre_landsat_" in fname and fname.endswith(".tif"):
                    has_landsat = True
                    break
            
            if has_landsat and not has_sentinel:
                landsat_only_count += 1
                print(f"  Fire ID {fire_id}: Landsat presente, Sentinel assente.")
            elif not has_landsat and not has_sentinel:
                print(f"  Fire ID {fire_id}: Nessun dato Landsat o Sentinel trovato.")
            elif has_landsat and has_sentinel:
                print(f"  Fire ID {fire_id}: Landsat e Sentinel presenti.")
            elif not has_landsat and has_sentinel:
                 print(f"  Fire ID {fire_id}: Sentinel presente, Landsat assente.")


    print(f"\n--- Riepilogo ---")
    print(f"Totale cartelle incendi analizzate: {total_fires}")
    print(f"Cartelle con Landsat ma senza Sentinel: {landsat_only_count}")
    print(f"Percentuale 'Landsat-only': { (landsat_only_count / total_fires * 100):.2f}%" if total_fires > 0 else "0.00%")
    
    return landsat_only_count

if __name__ == "__main__":
    # Assicurati che questo percorso sia corretto per la tua struttura
    ROOT_DATASET_FOLDER = "piedmont_new" 
    count_landsat_only_fires(ROOT_DATASET_FOLDER)