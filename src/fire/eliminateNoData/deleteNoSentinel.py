import os
import shutil

def delete_empty_sentinel_folders(root_folder: str):
    """
    Scorre tutte le cartelle degli incendi e rimuove quelle che non contengono
    almeno un'immagine Sentinel-2, riconoscibile dal nome del file.
    
    Args:
        root_folder (str): Il percorso della cartella radice (es. 'piedmont_new').
    """
    if not os.path.isdir(root_folder):
        print(f"ERRORE: La cartella '{root_folder}' non esiste. Verifica il percorso.")
        return

    print(f"Inizio la pulizia delle cartelle in '{root_folder}'...")

    folders_to_delete = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Controlla solo le cartelle "fire_*"
        if os.path.basename(dirpath).startswith("fire_"):
            
            # Controlla se esiste almeno un file Sentinel nella lista di tutti i files
            has_sentinel = any("pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f for f in filenames)
            
            # Se non ci sono file Sentinel, aggiungi la cartella alla lista di quelle da eliminare
            if not has_sentinel:
                folders_to_delete.append(dirpath)
    
    # Conferma prima di eliminare
    if folders_to_delete:
        print("\nLe seguenti cartelle verranno eliminate perché non contengono file Sentinel:")
        for folder in folders_to_delete:
            print(f"  - {folder}")
        
        confirmation = input("\nSei sicuro di voler procedere con l'eliminazione? (sì/no): ")
        if confirmation.lower() == 'sì' or confirmation.lower() == 'si':
            for folder in folders_to_delete:
                try:
                    shutil.rmtree(folder)
                    print(f"✅ Cartella eliminata: {folder}")
                except Exception as e:
                    print(f"❌ Errore durante l'eliminazione della cartella {folder}: {e}")
            print("\n🎉 Pulizia completata.")
        else:
            print("\nOperazione annullata.")
    else:
        print("\nNessuna cartella da eliminare trovata. Il dataset è già pulito.")

if __name__ == "__main__":
    # Configura qui il percorso della cartella radice del tuo dataset
    root_directory = "piedmont_new"
    delete_empty_sentinel_folders(root_directory)