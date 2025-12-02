import os
import rasterio
import numpy as np
from collections import defaultdict

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
                print(f"⚠️ Warning: Maschere con shape o CRS non corrispondenti per {os.path.basename(cm_mask_path)}. Saltato.")
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

def find_best_image_in_folder(fire_folder_path):
    """
    Trova la migliore immagine Sentinel in una cartella, basandosi sulla maschera delle nuvole.

    Args:
        fire_folder_path (str): Percorso della cartella di un singolo incendio (es. 'piedmont/fire_1234').

    Returns:
        dict: Dizionario con il percorso del file migliore e il suo punteggio,
              o None se non viene trovata un'immagine valida.
    """
    print(f"\nAnalisi cartella: {fire_folder_path}")
    
    # 1. Trova la GTMask per Sentinel
    gt_mask_sentinel_path = None
    for file in os.listdir(fire_folder_path):
        if file.endswith("GTSentinel.tif"):
            gt_mask_sentinel_path = os.path.join(fire_folder_path, file)
            break
            
    if not gt_mask_sentinel_path or not os.path.exists(gt_mask_sentinel_path):
        print("  → ⚠️ GTMask Sentinel non trovata. Ignorato.")
        return None

    # 2. Raccogli tutte le coppie Sentinel + Cloud Mask
    sentinel_cm_pairs = defaultdict(lambda: {'sentinel_path': None, 'cm_path': None})
    
    for file in os.listdir(fire_folder_path):
        # Cerchiamo i file Sentinel e le loro Cloud Mask
        # Escludiamo le GT, Landsat e altre CM che non ci interessano
        if file.endswith("_CM.tif") and "sentinel" in file and "GT" not in file:
            # Estrai il nome del file originale dalla CM (es. 'fire_..._sentinel.tif')
            original_filename = file.replace("_CM.tif", ".tif")
            sentinel_path = os.path.join(fire_folder_path, original_filename)
            
            if os.path.exists(sentinel_path):
                # Usa il nome originale come chiave per raggruppare
                sentinel_cm_pairs[original_filename]['cm_path'] = os.path.join(fire_folder_path, file)
                sentinel_cm_pairs[original_filename]['sentinel_path'] = sentinel_path

    # 3. Calcola il punteggio per ogni coppia valida
    image_scores = []
    for original_filename, paths in sentinel_cm_pairs.items():
        if paths['sentinel_path'] and paths['cm_path']:
            score = get_cloud_score_on_gt(gt_mask_sentinel_path, paths['cm_path'])
            if score is not None:
                image_scores.append({
                    'sentinel_path': paths['sentinel_path'],
                    'cm_path': paths['cm_path'],
                    'score': score
                })
                print(f"  → Calcolato punteggio per {os.path.basename(paths['sentinel_path'])}: {score}")

    # 4. Trova l'immagine con il punteggio più basso
    if not image_scores:
        print("  → ❌ Nessuna immagine Sentinel valida con CM trovata in questa cartella.")
        return None
        
    best_image = min(image_scores, key=lambda x: x['score'])
    
    print(f"\n  → ✨ Immagine migliore trovata: {os.path.basename(best_image['sentinel_path'])} con punteggio: {best_image['score']:.2f}")
    
    return best_image

# === ESECUZIONE PRINCIPALE ===
if __name__ == "__main__":
    # --- IMPOSTA LA CARTELLA DA ANALIZZARE QUI ---
    # Sostituisci 'fire_5438' con la cartella che ti interessa
    target_fire_folder_name = "fire_5028"
   
    # Percorso della cartella principale con tutte le sottocartelle degli incendi
    root_dir = "piedmont_new" 
    
    # Costruisci il percorso completo della cartella target
    target_folder_path = os.path.join(root_dir, target_fire_folder_name)
    
    # Verifica che la cartella esista
    if not os.path.isdir(target_folder_path):
        print(f"Errore: La cartella '{target_folder_path}' non esiste. Controlla il percorso.")
    else:
        # Trova la migliore immagine in quella cartella
        best_image_info = find_best_image_in_folder(target_folder_path)
        
        if best_image_info:
            print("\n-----------------------------")
            print("--- RIASSUNTO RISULTATI ---")
            print("-----------------------------")
            print(f"Folder Analizzato: {target_fire_folder_name}")
            print(f"Immagine Sentinel migliore: {os.path.basename(best_image_info['sentinel_path'])}")
            print(f"Percorso Cloud Mask: {best_image_info['cm_path']}")
            print(f"Punteggio di nuvolosità sulla fire area: {best_image_info['score']:.2f}")
            print("-----------------------------")
        else:
            print(f"\nNessuna immagine Sentinel valida trovata nella cartella '{target_fire_folder_name}'.")