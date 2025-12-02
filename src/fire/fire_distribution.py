import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def analyze_fire_size_distribution(root_folder: str):
    """
    Analizza la distribuzione del numero di pixel di incendio 
    (valore 1) in tutti i file GTSentinel.tif e salva un istogramma.
    
    Args:
        root_folder (str): Il percorso della cartella radice (es. 'piedmont_new').
    """
    if not os.path.isdir(root_folder):
        print(f"ERRORE: La cartella '{root_folder}' non esiste. Verifica il percorso.")
        return

    fire_sizes = {}
    total_processed_files = 0
    
    print(f"Inizio l'analisi della distribuzione delle dimensioni degli incendi in '{root_folder}'...")

    for dirpath, dirnames, filenames in os.walk(root_folder):
        if os.path.basename(dirpath).startswith("fire_"):
            fire_id = os.path.basename(dirpath).replace("fire_", "")
            gt_file_found = False
            
            for filename in filenames:
                if filename.endswith("GTSentinel.tif"):
                    gt_file_path = os.path.join(dirpath, filename)
                    gt_file_found = True
                    total_processed_files += 1
                    
                    try:
                        with rasterio.open(gt_file_path) as src:
                            gt_data = src.read(1)
                            burned_pixel_count = np.sum(gt_data == 1)
                            fire_sizes[fire_id] = burned_pixel_count
                            print(f"Analizzato incendio ID {fire_id}: {burned_pixel_count} pixel bruciati.")
                    except Exception as e:
                        print(f"❌ Errore durante l'apertura del file {gt_file_path}: {e}")
            
            if not gt_file_found:
                print(f"⚠️ ATTENZIONE: Nessun file GTSentinel.tif trovato nella cartella 'fire_{fire_id}'.")

    if not fire_sizes:
        print("\nNessun dato di incendio trovato o analizzato. Verifica i nomi dei file e delle cartelle.")
        return

    print("\n--- Riepilogo Analisi ---")
    
    # Ordina gli incendi per dimensione, dal più piccolo al più grande
    sorted_fire_sizes = sorted(fire_sizes.items(), key=lambda item: item[1])
    
    print(f"Numero totale di incendi analizzati: {len(fire_sizes)}")
    print(f"Incendio più piccolo (ID {sorted_fire_sizes[0][0]}): {sorted_fire_sizes[0][1]} pixel")
    print(f"Incendio più grande (ID {sorted_fire_sizes[-1][0]}): {sorted_fire_sizes[-1][1]} pixel")
    
    # Calcola e stampa la media e la deviazione standard
    pixel_counts = list(fire_sizes.values())
    avg_size = np.mean(pixel_counts)
    std_dev = np.std(pixel_counts)
    print(f"Dimensione media degli incendi: {avg_size:.2f} pixel")
    print(f"Deviazione standard: {std_dev:.2f} pixel")
    
    # Genera un istogramma e lo salva come file PNG
    print("\nCreazione e salvataggio dell'istogramma...")
    plt.figure(figsize=(10, 6))
    plt.hist(pixel_counts, bins=20, edgecolor='black')
    plt.title('Fire Size Distribution  (Pixel Count)')
    plt.xlabel('Burning Pixel Count')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # Salva il plot come PNG invece di mostrarlo
    plt.savefig('fire_size_distribution.png')
    plt.close() # Chiude la figura per liberare memoria
    print("🎉 Istogramma salvato con successo nel file: fire_size_distribution.png")

if __name__ == "__main__":
    # Inserisci qui il percorso della tua cartella radice
    root_directory = "piedmont_new"
    analyze_fire_size_distribution(root_directory)