import numpy as np
import os
import rasterio
from collections import defaultdict
from rasterio.warp import reproject, Resampling
from typing import Tuple
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def resample_image(
    input_tif_path: str,
    output_tif_path: str,
    target_transform: Tuple,
    target_crs: rasterio.crs.CRS,
    target_shape: Tuple[int, int],
    resampling_method: Resampling = Resampling.bilinear
) -> None:
    """
    Resamples a multi-band raster image to a target resolution and grid
    and copies band descriptions to the output file.

    Args:
        input_tif_path (str): Path to the input TIFF file.
        output_tif_path (str): Path to save the resampled output TIFF file.
        target_transform (tuple): The affine transform of the target raster.
        target_crs (rasterio.crs.CRS): The CRS of the target raster.
        target_shape (tuple): The shape (height, width) of the target raster.
        resampling_method (rasterio.warp.Resampling): The resampling algorithm to use.
    """
    with rasterio.open(input_tif_path) as src:
        # Prepare an array to hold the resampled data for all bands
        resampled_data = np.empty(
            (src.count, target_shape[0], target_shape[1]),
            dtype=src.dtypes[0]
        )

        # Reproject each band
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=resampled_data[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_method
            )

        # Get the original band descriptions
        band_descriptions = [src.descriptions[i] if src.descriptions is not None and i < len(src.descriptions) else f"Band_{i+1}" for i in range(src.count)]

        # Update the profile for the output file
        profile = src.profile
        profile.update(
            width=target_shape[1],
            height=target_shape[0],
            transform=target_transform,
            crs=target_crs,
            nodata=src.nodata,
            dtype=resampled_data.dtype,
            count=src.count # Ensure band count is correct
        )
        
        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(resampled_data)
            # Copy band descriptions from source to destination
            for i, desc in enumerate(band_descriptions):
                if desc is not None:
                    dst.set_band_description(i + 1, desc)

def read_image_and_metadata(image_path: str):
    """
    Reads a raster image and its key metadata (transform, CRS, shape, resolution, band descriptions).
    """
    with rasterio.open(image_path) as src:
        img_data = src.read() # C x H x W
        resolution = (src.res[0], src.res[1])
        # Read band descriptions, defaulting to a generic name if not available
        band_descriptions = src.descriptions if src.descriptions is not None else [f"Band_{i+1}" for i in range(src.count)]
        return img_data, src.transform, src.crs, src.shape, resolution, band_descriptions



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
                #print(f"⚠️ Warning: Maschere con shape o CRS non corrispondenti per {os.path.basename(cm_mask_path)}. Saltato.")
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
    #print(f"\nAnalisi cartella: {fire_folder_path}")
    
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
                #print(f"  → Calcolato punteggio per {os.path.basename(paths['sentinel_path'])}: {score}")

    # 4. Trova l'immagine con il punteggio più basso
    if not image_scores:
        print("  → ❌ Nessuna immagine Sentinel valida con CM trovata in questa cartella.")
        return None
        
    best_image = min(image_scores, key=lambda x: x['score'])
    
    #print(f"\n  → ✨ Immagine migliore trovata: {os.path.basename(best_image['sentinel_path'])} con punteggio: {best_image['score']:.2f}")
    
    return best_image

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr

def fire_area_iou(hist):
    true_postives = hist[1,1]
    false_positives = hist[0,1]
    false_negatives = hist[1,0]
    true_negatives = hist[0,0]
    iou = true_postives / (true_postives + false_positives + false_negatives)
    return iou

def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

