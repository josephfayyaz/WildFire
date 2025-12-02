import os

# Configurazione della cartella radice del dataset
ROOT_DATASET_FOLDER = "piedmont_new" # Assicurati che questo percorso sia corretto

def delete_all_ignition_tifs(root_folder: str):
    """
    Cerca e cancella tutti i file *_ignition_point.tif all'interno di ogni sottocartella fire_XXX
    nella cartella radice specificata.
    """
    print(f"Cancellazione di tutti i file *_ignition_point.tif nella cartella: {root_folder}")
    deleted_count = 0
    
    if not os.path.exists(root_folder):
        print(f"ERRORE: La cartella '{root_folder}' non esiste. Impossibile procedere.")
        return

    for item in os.listdir(root_folder):
        fire_folder_path = os.path.join(root_folder, item)
        
        # Controlla se è una cartella e se il nome inizia con "fire_"
        if os.path.isdir(fire_folder_path) and item.startswith("fire_"):
            # Costruisci il nome del file ignition point atteso
            fire_id = item.replace("fire_", "")
            ignition_file_name = f"fire_{fire_id}_ignition_pt.tif"
            ignition_file_path = os.path.join(fire_folder_path, ignition_file_name)
            
            # Se il file esiste, cancellalo
            if os.path.exists(ignition_file_path):
                try:
                    os.remove(ignition_file_path)
                    print(f"  Cancellato: {ignition_file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ERRORE: Impossibile cancellare '{ignition_file_path}': {e}")
            # else:
            #     print(f"  Non trovato: {ignition_file_path}") # Utile per debug, ma può essere verboso
    
    print(f"\nCancellazione completata. Totale file *_ignition_point.tif cancellati: {deleted_count}")

if __name__ == "__main__":
    confirm = input(f"Sei sicuro di voler cancellare TUTTI i file *_ignition_point.tif in '{ROOT_DATASET_FOLDER}' e nelle sue sottocartelle? Digita 'si' per continuare: ")
    if confirm.lower() == 'si':
        delete_all_ignition_tifs(ROOT_DATASET_FOLDER)
    else:
        print("Operazione annullata.")