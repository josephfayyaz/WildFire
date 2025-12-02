import os
import numpy as np
import rasterio
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- CONFIGURAZIONE ---
DATA_ROOT = 'piedmont_new'

# Soglie per il filtro
MIN_PIXEL_COUNT = 10  # Numero minimo di pixel bruciati per essere considerato "gigantesco"
MAX_DECENTER_DISTANCE_PX = 100  # Distanza massima (in pixel) dal centro dell'immagine

# --- FUNZIONI DI ANALISI ---
def get_all_fire_dirs_and_gt_paths(root_dir: str) -> List[Tuple[str, str]]:
    """
    Ottiene un elenco di tutte le directory di incendio che contengono una GT mask,
    restituendo una tupla con il nome della cartella e il percorso completo del file.
    """
    fire_info = []
    for d in os.listdir(root_dir):
        full_path = os.path.join(root_dir, d)
        if os.path.isdir(full_path):
            gt_file = None
            for f in os.listdir(full_path):
                # Cerca il file con estensione .tif e che contenga "GTSentinel"
                if f.endswith('.tif') and 'GTSentinel' in f:
                    gt_file = f
                    break
            
            if gt_file:
                gt_path = os.path.join(full_path, gt_file)
                fire_info.append((d, gt_path))
                
    return sorted(fire_info)

def analyze_gt_masks(all_fire_info: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analizza le maschere di ground truth per identificare aree gigantesche o decentrate.
    Restituisce un dizionario con due liste di tuple: (nome_folder, valore_analizzato).
    """
    problematic_folders = {
        'gigantic_area': [],
        'decentered_area': []
    }

    for fire_dir, gt_path in tqdm(all_fire_info, desc="Analyzing GT masks"):
        try:
            with rasterio.open(gt_path) as src:
                gt_mask = src.read(1)
            
            # 1. Analisi dell'area (conteggio dei pixel)
            pixel_count = np.sum(gt_mask == 1)
            if pixel_count < MIN_PIXEL_COUNT:
                problematic_folders['gigantic_area'].append((fire_dir, pixel_count))

            # 2. Analisi della posizione (decentramento)
            if pixel_count > 0:
                # Trova gli indici dei pixel bruciati
                burned_pixels_coords = np.argwhere(gt_mask == 1)
                
                # Calcola il centroide dei pixel bruciati
                centroid_y = burned_pixels_coords[:, 0].mean()
                centroid_x = burned_pixels_coords[:, 1].mean()
                
                # Calcola il centro dell'immagine
                image_center_y = gt_mask.shape[0] / 2
                image_center_x = gt_mask.shape[1] / 2

                # Calcola la distanza euclidea dal centro dell'immagine
                distance = np.sqrt((centroid_x - image_center_x)**2 + (centroid_y - image_center_y)**2)
                
                if distance > MAX_DECENTER_DISTANCE_PX:
                    problematic_folders['decentered_area'].append((fire_dir, distance))

        except Exception as e:
            print(f"Errore nell'analisi della maschera per {fire_dir}: {e}")
            continue
            
    return problematic_folders

# --- ESECUZIONE ---
def main():
    print("--- Avvio dell'analisi delle maschere di ground truth ---")
    print(f"Soglia per aree gigantesche (pixel): > {MIN_PIXEL_COUNT}")
    print(f"Soglia per decentramento (pixel): > {MAX_DECENTER_DISTANCE_PX} px")
    
    all_info = get_all_fire_dirs_and_gt_paths(DATA_ROOT)
    print(f"Trovati {len(all_info)} folder con maschere GT.")
    
    problematic = analyze_gt_masks(all_info)
    
    print("\n--- Aree con numero di pixel bruciati molto elevato ---")
    if problematic['gigantic_area']:
        for folder, pixel_count in problematic['gigantic_area']:
            print(f"  - {folder}: {pixel_count} pixel")
    else:
        print("Nessuna area trovata che superi la soglia.")

    print("\n--- Aree bruciate troppo decentrate ---")
    if problematic['decentered_area']:
        for folder, distance in problematic['decentered_area']:
            print(f"  - {folder}: Distanza dal centro {distance:.2f} px")
    else:
        print("Nessuna area trovata che superi la soglia di decentramento.")

    print("\n--- Analisi completata ---")

if __name__ == '__main__':
    main()
