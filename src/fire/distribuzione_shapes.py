import os
import rasterio
import re
from typing import List, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_image_shapes(root_dir: str) -> Dict[Tuple[int, int], int]:
    """
    Analizza le dimensioni (H, W) di tutte le immagini Sentinel-2 post-fire
    (quelle che non contengono 'pre_sentinel' nel nome ma 'sentinel' e '.tif')
    presenti nelle sottocartelle di root_dir.
    Conta le occorrenze di ogni shape.
    """
    shape_counts = Counter()
    processed_files = 0
    skipped_files = 0

    all_fire_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, d))]

    print(f"Inizio analisi delle shape in {root_dir}...")

    for fire_dir in all_fire_dirs:
        files_in_dir = os.listdir(fire_dir)
        
        # Cerca solo le immagini post-sentinel, escludendo _CM e pre_sentinel
        # Prendi la prima trovata, assumendo che le immagini all'interno di una cartella abbiano la stessa shape.
        target_image_file = next((
            f for f in files_in_dir
            if f.endswith(".tif") and "pre_sentinel" in f and "_CM" not in f
        ), None)

        if target_image_file:
            image_path = os.path.join(fire_dir, target_image_file)
            try:
                with rasterio.open(image_path) as src:
                    shape = src.shape  # Restituisce (height, width)
                    shape_counts[shape] += 1
                    processed_files += 1
            except Exception as e:
                print(f"Errore durante la lettura di {image_path}: {e}")
                skipped_files += 1
        else:
            skipped_files += 1

    print(f"\nAnalisi completata. Processati {processed_files} immagini, saltati {skipped_files} directory/file.")
    return shape_counts

def plot_shape_distribution(shape_counts: Dict[Tuple[int, int], int], min_display_count: int = 5):
    """
    Genera un istogramma della distribuzione delle larghezze e altezze,
    e stampa le shape più comuni.
    """
    if not shape_counts:
        print("Nessuna shape trovata per la visualizzazione.")
        return

    heights = [s[0] for s in shape_counts.keys()]
    widths = [s[1] for s in shape_counts.keys()]
    counts = list(shape_counts.values())

    print("\n--- Distribuzione delle Shape (Occorrenze) ---")
    sorted_shapes = sorted(shape_counts.items(), key=lambda item: item[1], reverse=True)
    for shape, count in sorted_shapes:
        print(f"  Shape {shape[0]}x{shape[1]}: {count} occorrenze")

    # Filtra per mostrare solo le shape con almeno min_display_count occorrenze nel grafico
    filtered_heights = []
    filtered_widths = []
    filtered_counts = []
    for (h, w), count in sorted_shapes:
        if count >= min_display_count:
            filtered_heights.append(h)
            filtered_widths.append(w)
            filtered_counts.append(count)

    if not filtered_heights:
        print(f"\nNessuna shape con almeno {min_display_count} occorrenze da plottare.")
        return

    # Visualizzazione
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Istogramma delle altezze
    axes[0].hist(heights, bins=np.arange(min(heights), max(heights) + 2, 1), edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribuzione Altezze Immagini')
    axes[0].set_xlabel('Altezza (pixel)')
    axes[0].set_ylabel('Frequenza')
    axes[0].grid(axis='y', alpha=0.75)
    axes[0].set_xticks(np.arange(min(heights), max(heights) + 1, max(1, (max(heights) - min(heights)) // 10))) # Aggiusta i tick

    # Istogramma delle larghezze
    axes[1].hist(widths, bins=np.arange(min(widths), max(widths) + 2, 1), edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribuzione Larghezze Immagini')
    axes[1].set_xlabel('Larghezza (pixel)')
    axes[1].set_ylabel('Frequenza')
    axes[1].grid(axis='y', alpha=0.75)
    axes[1].set_xticks(np.arange(min(widths), max(widths) + 1, max(1, (max(widths) - min(widths)) // 10))) # Aggiusta i tick

    plt.tight_layout()
    plt.show()
    plt.savefig('shape_distribution.png', dpi=300)

if __name__ == "__main__":
    # Sostituisci con il percorso alla tua directory radice contenente le cartelle 'fire_XXXX'
    # Esempio: ROOT_DATA_DIR = '/home/bavaro/data/piedmont_dataset_prepared'
    # O se sei nella stessa directory del dataset: ROOT_DATA_DIR = '.'
    ROOT_DATA_DIR = 'piedmont_new' # <--- CAMBIA QUESTO PERCORSO!

    
    shapes = analyze_image_shapes(ROOT_DATA_DIR)
    plot_shape_distribution(shapes, min_display_count=5) # Puoi cambiare min_display_count