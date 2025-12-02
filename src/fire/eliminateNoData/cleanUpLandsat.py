import os
import rasterio
import shutil

def delete_landsat_files_with_wrong_bands(root_folder: str, expected_bands: int = 15):
    """
    Scorre la cartella radice e tutte le sue sottocartelle per trovare
    e rimuovere i file Landsat che non hanno il numero di bande atteso.

    Args:
        root_folder (str): Il percorso della cartella radice del dataset.
        expected_bands (int): Il numero di bande atteso per i file Landsat.
    """
    if not os.path.isdir(root_folder):
        print(f"ERRORE: La cartella '{root_folder}' non esiste. Verifica il percorso.")
        return

    print(f"Inizio la ricerca di file Landsat con meno di {expected_bands} bande in '{root_folder}'...")

    files_to_delete = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # Controlla se il file è un'immagine Landsat e ha l'estensione .tif
            if "_pre_landsat" in filename and filename.endswith(".tif"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with rasterio.open(file_path) as src:
                        num_bands = src.count
                        if num_bands < expected_bands:
                            print(f"⚠️ Trovato file non valido: '{file_path}' (Bande: {num_bands}, attese: {expected_bands})")
                            files_to_delete.append(file_path)
                except Exception as e:
                    # Gestisce errori di apertura del file (es. file corrotto)
                    print(f"❌ Errore nell'apertura del file {file_path}: {e}")
                    files_to_delete.append(file_path)

    # Conferma prima di eliminare
    if files_to_delete:
        print("\n--- Riepilogo ---")
        print(f"Numero totale di file da eliminare: {len(files_to_delete)}")
        print("I seguenti file verranno rimossi:")
        for file_path in files_to_delete:
            print(f"  - {file_path}")
        
        confirmation = input("\nSei sicuro di voler procedere con l'eliminazione? (sì/no): ")
        if confirmation.lower() == 'sì' or confirmation.lower() == 'si':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"✅ File eliminato: {file_path}")
                except Exception as e:
                    print(f"❌ Errore durante l'eliminazione del file {file_path}: {e}")
            print("\n🎉 Pulizia completata.")
        else:
            print("\nOperazione annullata.")
    else:
        print("\nNessun file da eliminare trovato. Il dataset è già pulito.")

if __name__ == "__main__":
    # Configura qui il percorso della cartella radice del tuo dataset
    root_directory = "piedmont_new"
    
    # Esegui la funzione con il percorso corretto
    delete_landsat_files_with_wrong_bands(root_directory, expected_bands=15)