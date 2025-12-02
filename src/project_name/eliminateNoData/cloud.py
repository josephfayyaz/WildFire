import os
import glob
import rasterio
import numpy as np
import shutil

# Aggiungi qui il percorso della tua cartella principale del dataset.
# Esempio: "/home/user/dataset_incendi"
DATASET_ROOT_PATH = "piedmont_new"

def get_cloud_score_on_gt(gt_mask_path, cm_mask_path):
    """
    Calcola il 'cloud score' sommando la GTMask e la Cloud Mask.
    Il punteggio è la somma dei valori dei pixel della CM solo dove la GT è 1.
    
    Args:
        gt_mask_path (str): Percorso del file TIFF della maschera GT.
        cm_mask_path (str): Percorso del file TIFF della maschera delle nuvole (CM).
        
    Returns:
        float: Il punteggio totale di nuvolosità sulla fire area, o None in caso di errore.
    """
    try:
        with rasterio.open(gt_mask_path) as gt_src, rasterio.open(cm_mask_path) as cm_src:
            # Assicurati che le due immagini abbiano la stessa forma e CRS
            if gt_src.shape != cm_src.shape or gt_src.crs != cm_src.crs:
                return None
            
            # Leggi i dati della maschera GT e della CM
            gt_mask = gt_src.read(1)
            cm_mask = cm_src.read(1)
            
            # Crea una maschera booleana per la fire area (dove GT è 1)
            fire_area_mask = (gt_mask == 1)
            
            # Applica la maschera alla CM per ottenere solo i valori sulla fire area
            cm_on_fire_area = cm_mask[fire_area_mask]
            
            # Calcola il punteggio: somma dei valori della CM sulla fire area
            # I valori della CM sono: 0 (Clear), 1 (Thick Cloud), 2 (Thin Cloud), 3 (Shadow)
            # Un punteggio più basso è migliore.
            cloud_score = np.sum(cm_on_fire_area)
            
            return float(cloud_score)
            
    except rasterio.errors.RasterioIOError as e:
        print(f"❌ Errore di I/O con un file, probabilmente corrotto o mancante: {e}")
        return None
    except Exception as e:
        print(f"❌ Errore inaspettato durante il calcolo dello score: {e}")
        return None

def find_single_sentinel_folders(root_dir):
    """
    Scorre tutte le sottocartelle per trovare quelle che contengono un'unica
    immagine Sentinel e ne calcola il cloud score.

    Args:
        root_dir (str): La cartella radice del dataset.
    """
    if not os.path.isdir(root_dir):
        print(f"ERRORE: La cartella '{root_dir}' non esiste. Verifica il percorso.")
        return

    print(f"Inizio la scansione per cartelle con una singola immagine Sentinel in '{root_dir}'...")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Controlla solo le cartelle "fire_*"
        if os.path.basename(dirpath).startswith("fire_"):
            
            # Cerca i file Sentinel, ignorando le maschere di nuvole (CM)
            sentinel_images = sorted([f for f in filenames if "pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f])
            
            # Se la cartella contiene esattamente un'immagine Sentinel
            if len(sentinel_images) == 1:
                folder_name = os.path.basename(dirpath)
                
                # Ottieni il nome del file Sentinel senza estensione
                base_name = os.path.splitext(sentinel_images[0])[0]
                
                # Cerca i percorsi delle maschere GT e CM basandoti sul nome del file
                # Assumiamo che la maschera CM abbia lo stesso nome del file Sentinel ma con "_CM.tif"
                # E che la maschera GT abbia il prefisso "GTSentinel"
                cm_mask_path = os.path.join(dirpath, f"{base_name}_CM.tif") 
                gt_mask_path = os.path.join(dirpath, f"{folder_name}_GTSentinel.tif")

                if not os.path.exists(gt_mask_path) or not os.path.exists(cm_mask_path):
                    print(f"⚠️ Warning: File di maschera non trovati per la cartella {folder_name}. Saltato.")
                    continue

                # Calcola il cloud score usando la funzione fornita
                score = get_cloud_score_on_gt(gt_mask_path, cm_mask_path)
                
                if score is not None:
                    print(f"✔️ Cartella: {folder_name} -> Cloud Score: {score}")
                else:
                    print(f"❌ Impossibile calcolare il Cloud Score per la cartella {folder_name}. Controlla i file.")
            
    print("\nScansione completata.")

if __name__ == "__main__":
    find_single_sentinel_folders(DATASET_ROOT_PATH)
