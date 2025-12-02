import planetary_computer
import pystac_client
import geopandas as gpd
from shapely.geometry import shape, box
import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
import yaml
from datetime import datetime, timedelta
import pandas as pd
from rasterio.features import rasterize
from PIL import Image

# Funzioni di supporto
def load_config(config_path="src/project_name/config.yaml"):
    """Carica il file di configurazione YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_gt_as_colored_png(gt_mask_array, png_path):
    """
    Salva la maschera GT (array numpy 0/1) come immagine PNG colorata.
    I pixel con valore 1 saranno rossi, gli altri neri.
    """
    rgb = np.zeros((gt_mask_array.shape[0], gt_mask_array.shape[1], 3), dtype=np.uint8)
    rgb[gt_mask_array == 1] = [255, 0, 0]  # Red
    img = Image.fromarray(rgb)
    img.save(png_path)

def are_images_similar(img1, img2, tolerance_percentage=0.005):
    """
    Compara due array NumPy (immagini) per verificarne la somiglianza.
    Ritorna True se la differenza relativa media per pixel è inferiore alla tolleranza.
    img1 e img2 devono avere la stessa shape e dtype.
    """
    if img1.shape != img2.shape or img1.dtype != img2.dtype:
        return False 

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    max_val = np.iinfo(img1.dtype).max if np.issubdtype(img1.dtype, np.integer) else 1.0 
    
    mean_relative_diff = np.mean(diff / max_val)
    
    return mean_relative_diff < tolerance_percentage


def process_single_fire(fire_id, config):
    """
    Processa un singolo incendio, cercando immagini Sentinel-2 *prima* dell'incendio,
    ritagliandole e generando una singola maschera GT per incendio.
    Aggiunto controllo di somiglianza per immagini dello stesso giorno.
    """
    
    main_geojson_path = config["geojson_path"]
    ROOT_DATASET_FOLDER = "piedmont_new" 
    
    os.makedirs(ROOT_DATASET_FOLDER, exist_ok=True)
    
    print(f"Caricamento GeoJSON da: {main_geojson_path}")
    gdf_all_fires = gpd.read_file(main_geojson_path)
    
    # AGGIORNAMENTO: Leggi target_crs dal config
    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632") 
    if str(gdf_all_fires.crs) != TARGET_CRS_FOR_FIRES:
        print(f"Convertendo GeoJSON da {gdf_all_fires.crs} a {TARGET_CRS_FOR_FIRES} per i calcoli interni.")
        gdf_all_fires = gdf_all_fires.to_crs(TARGET_CRS_FOR_FIRES)

    fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
    
    if fire_data.empty:
        print(f"❌ Incendio con ID {fire_id} non trovato nel GeoJSON.")
        return

    fire = fire_data.iloc[0]
    fire_date = pd.to_datetime(fire["initialdate"])
    gt_geometry_utm = shape(fire["geometry"])
    
    # AGGIORNAMENTO: Leggi min_fire_area_sq_m dal config
    min_fire_area_sq_m = config.get("min_fire_area_sq_m", 200) 
    if gt_geometry_utm.area < min_fire_area_sq_m:
        print(f"❌ Incendio ID {fire_id} ha un'area troppo piccola ({gt_geometry_utm.area:.2f} mq) per essere utile a 10m di risoluzione. Minimo richiesto: {min_fire_area_sq_m} mq. Saltato.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    TARGET_RESOLUTION_M = 10 
    patch_size_pixels = config.get("patch_size_pixels", 256)
    # AGGIORNAMENTO: Leggi interval_pre_fire_days dal config
    interval_days = config.get("interval_pre_fire_days", 7) 
    # AGGIORNAMENTO: Leggi image_similarity_tolerance dal config
    image_similarity_tolerance = config.get("image_similarity_tolerance", 0.005)


    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M 

    FIRE_SAVE_FOLDER = os.path.join(ROOT_DATASET_FOLDER, f"fire_{fire_id}")
    os.makedirs(FIRE_SAVE_FOLDER, exist_ok=True)

    buffer_for_stac_search_meters = fixed_patch_size_meters / 2 
    fire_geometry_search_buffer_utm = gt_geometry_utm.buffer(buffer_for_stac_search_meters)
    fire_geometry_search_buffer_wgs84 = gpd.GeoSeries([fire_geometry_search_buffer_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(epsg=4326).iloc[0]

    time_range_start = fire_date - timedelta(days=interval_days) 
    time_range_end = fire_date 
    time_range_str = f"{time_range_start.strftime('%Y-%m-%d')}/{time_range_end.strftime('%Y-%m-%d')}"

    print(f"\n--- Processando Incendio ID: {fire_id} ---")
    print("Data dell'incendio:", fire_date.strftime('%Y-%m-%d'))
    print("Intervallo di ricerca STAC (PRE-incendio):", time_range_str)
    print(f"Dimensione patch target (finale): {patch_size_pixels}x{patch_size_pixels} pixel ({fixed_patch_size_meters/1000:.2f} km per lato).")
    print(f"Cartella di output: {FIRE_SAVE_FOLDER}")
    print(f"Tolleranza di somiglianza immagini: {image_similarity_tolerance*100:.3f}%")


    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    min_date_for_sentinel2 = datetime(2015, 6, 23) 
    if time_range_end < min_date_for_sentinel2: 
        print(f"❌ Incendio ID {fire_id} (intervallo termina il {time_range_end.strftime('%Y-%m-%d')}) è precedente alla disponibilità di Sentinel-2 ({min_date_for_sentinel2.strftime('%Y-%m-%d')}). Saltato.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    print(f"Ricerca STAC con bbox (WGS84): {fire_geometry_search_buffer_wgs84.bounds}")
    
    # AGGIORNAMENTO: Verifica il valore di 'satellite' dal config
    collection_name = config.get("satellite", "sentinel-2")
    if collection_name == "sentinel-2":
        stac_collections = ["sentinel-2-l2a"]
    # Qui potresti aggiungere 'elif' per altri satelliti se in futuro li supporterai
    else:
        print(f"⚠️ Collezione STAC per satellite '{collection_name}' non riconosciuta. Usando 'sentinel-2-l2a' come default.")
        stac_collections = ["sentinel-2-l2a"]


    search = catalog.search(
        collections=stac_collections,
        bbox=fire_geometry_search_buffer_wgs84.bounds,
        datetime=time_range_str,
        limit=500 
    )

    items = list(search.items())
    if not items:
        print(f"❌ Nessuna immagine {collection_name.upper()} trovata per la regione/data nell'intervallo PRE-incendio.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    print(f"✅ Trovate {len(items)} immagini {collection_name.upper()} nell'intervallo PRE-incendio.")

    sentinel2_bands = {
        "B02": 10, "B03": 10, "B04": 10, "B08": 10,
        "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
        "B01": 60, "B09": 60
    }
    
    valid_items = []
    
    for i, item_candidate in enumerate(items):
        ref_band_name = "B04"
        if ref_band_name not in item_candidate.assets:
             ref_band_name = next((b for b, r in sentinel2_bands.items() if r == 10 and b in item_candidate.assets), None)
        
        if ref_band_name is None:
            print(f"  Saltata immagine {i+1} del {item_candidate.datetime.strftime('%Y-%m-%d')} - Nessuna banda a 10m disponibile.")
            continue

        signed_href = planetary_computer.sign(item_candidate.assets[ref_band_name]).href
        try:
            with rasterio.open(signed_href) as src:
                item_bounds_polygon = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).iloc[0]
                gt_geometry_in_item_crs = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(src.crs).iloc[0]

                if gt_geometry_in_item_crs.intersects(item_bounds_polygon):
                    print(f"  Immagine {i+1} del {item_candidate.datetime.strftime('%Y-%m-%d')} interseca la GT. Aggiunta alla lista.")
                    valid_items.append(item_candidate)
                else:
                    print(f"  ❌ Immagine {i+1} del {item_candidate.datetime.strftime('%Y-%m-%d')} - Non interseca la geometria GT.")

        except Exception as e:
            print(f"  Errore durante l'apertura/lettura dell'immagine {i+1}: {e}")

    if not valid_items:
        print("🔴 Nessuna delle immagini Sentinel-2 trovate interseca la geometria dell'incendio o ha bande valide. Impossibile procedere con il ritaglio.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    print(f"\n✅ Iniziamo l'elaborazione di {len(valid_items)} immagini valide.")

    processed_images_by_date = {} 
    
    gt_generated_for_fire = False
    
    if not hasattr(process_single_fire, 'processed_fire_counts'):
        process_single_fire.processed_fire_counts = {}
    if fire_id not in process_single_fire.processed_fire_counts:
        process_single_fire.processed_fire_counts[fire_id] = 0

    for item_idx, item in enumerate(valid_items):
        current_date_str = item.datetime.strftime('%Y-%m-%d')
        
        print(f"\n📸 Elaborazione immagine {item_idx + 1}/{len(valid_items)}: {item.id} del {current_date_str}.")
        
        band_arrays_resampled = []
        band_names_resampled = []
        
        reference_band_name = "B04"
        if reference_band_name not in item.assets:
             reference_band_name = next((b for b, r in sentinel2_bands.items() if r == 10 and b in item.assets), None)

        if reference_band_name is None:
            print("⚠️ Nessuna banda a 10m trovata per il riferimento nell'immagine corrente. Skip questa immagine.")
            continue
        
        signed_href = planetary_computer.sign(item.assets[reference_band_name]).href
        with rasterio.open(signed_href) as ref_src:
            if str(ref_src.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES: 
                 print(f"🔴 CRITICAL WARNING: Il CRS del GeoJSON ({TARGET_CRS_FOR_FIRES}) NON corrisponde al CRS dell'immagine Sentinel ({ref_src.crs})! Questo causerà disallineamenti.")
            
            if str(ref_src.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                gt_geometry_in_ref_src_crs = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(ref_src.crs).iloc[0]
            else:
                gt_geometry_in_ref_src_crs = gt_geometry_utm
            
            centroid_x, centroid_y = gt_geometry_in_ref_src_crs.centroid.x, gt_geometry_in_ref_src_crs.centroid.y
            
            minx_fixed_patch = centroid_x - (fixed_patch_size_meters / 2)
            miny_fixed_patch = centroid_y - (fixed_patch_size_meters / 2)
            maxx_fixed_patch = centroid_x + (fixed_patch_size_meters / 2)
            maxy_fixed_patch = centroid_y + (fixed_patch_size_meters / 2)
            
            fixed_patch_bounds = (minx_fixed_patch, miny_fixed_patch, maxx_fixed_patch, maxy_fixed_patch)

            window_to_read = rasterio.windows.from_bounds(*fixed_patch_bounds, transform=ref_src.transform)

            window_to_read = window_to_read.intersection(rasterio.windows.Window(0, 0, ref_src.width, ref_src.height))
            window_to_read = window_to_read.round_offsets(op='floor').round_lengths(op='ceil')

            if window_to_read.width == 0 or window_to_read.height == 0:
                print("🔴 ERRORE: La finestra di lettura calcolata è vuota o invalida dopo l'intersezione con i limiti dell'immagine Sentinel. Incendio forse troppo vicino al bordo della tile o dati mancanti. Skip questa immagine.")
                continue
            
            final_transform = rasterio.transform.from_bounds(
                *rasterio.windows.bounds(window_to_read, ref_src.transform), 
                width=patch_size_pixels, 
                height=patch_size_pixels
            )

            output_profile = ref_src.profile.copy()
            output_profile.update({
                "height": patch_size_pixels,
                "width": patch_size_pixels,
                "transform": final_transform,
                "count": len(sentinel2_bands),
                "dtype": np.uint16 
            })

            bands_10m_for_comparison = [] 

            for band, native_resolution in sentinel2_bands.items():
                if band not in item.assets:
                    continue
                signed_href = planetary_computer.sign(item.assets[band]).href
                with rasterio.open(signed_href) as src:
                    try:
                        if native_resolution != TARGET_RESOLUTION_M:
                            scale_factor = native_resolution / TARGET_RESOLUTION_M
                            scaled_window = rasterio.windows.Window(
                                col_off=window_to_read.col_off / scale_factor,
                                row_off=window_to_read.row_off / scale_factor,
                                width=window_to_read.width / scale_factor,
                                height=window_to_read.height / scale_factor
                            )
                            scaled_window = scaled_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                            scaled_window = scaled_window.round_offsets(op='floor').round_lengths(op='ceil')
                        else:
                            scaled_window = window_to_read
                        
                        if scaled_window.width == 0 or scaled_window.height == 0:
                            print(f"  ⚠️ Banda {band}: window scalata invalida. Skip.")
                            continue

                        band_data = src.read(
                            1,
                            window=scaled_window,
                            out_shape=(patch_size_pixels, patch_size_pixels),
                            resampling=Resampling.bilinear
                        )
                        
                        if band_data.shape != (patch_size_pixels, patch_size_pixels):
                            print(f"  ⚠️ Banda {band} ha shape {band_data.shape}, attesa {(patch_size_pixels, patch_size_pixels)} dopo resize. Skip.")
                            continue
                        
                        band_arrays_resampled.append(band_data)
                        band_names_resampled.append(band)

                        if native_resolution == 10: 
                            bands_10m_for_comparison.append(band_data)

                    except Exception as e:
                        print(f"  ❌ Errore su banda {band} durante la lettura/resampling: {e}")

            if not band_arrays_resampled:
                print("⚠️ Nessuna banda utile trovata per questa patch. Impossibile salvare immagine Sentinel.")
                continue 

            data_cube = np.stack(band_arrays_resampled, axis=0)

            # --- Controllo di Somiglianza ---
            if current_date_str in processed_images_by_date:
                current_10m_bands_stack = np.stack(bands_10m_for_comparison, axis=0) if bands_10m_for_comparison else None
                
                is_similar_to_existing = False
                for existing_10m_bands_stack in processed_images_by_date[current_date_str]:
                    # AGGIORNAMENTO: Usa image_similarity_tolerance dal config
                    if current_10m_bands_stack is not None and are_images_similar(current_10m_bands_stack, existing_10m_bands_stack, tolerance_percentage=image_similarity_tolerance): 
                        is_similar_to_existing = True
                        break
                
                if is_similar_to_existing:
                    print(f"ℹ️ Immagine del {current_date_str} è troppo simile a una già processata. Saltata per evitare duplicati.")
                    continue
                else:
                    processed_images_by_date[current_date_str].append(current_10m_bands_stack)
            else:
                if bands_10m_for_comparison:
                    processed_images_by_date[current_date_str] = [np.stack(bands_10m_for_comparison, axis=0)]
                else:
                    processed_images_by_date[current_date_str] = []

            # --- Fine Controllo Somiglianza ---
            
            # Incrementa il contatore solo per le immagini uniche che vengono effettivamente salvate
            process_single_fire.processed_fire_counts[fire_id] += 1
            image_idx_in_fire = process_single_fire.processed_fire_counts[fire_id]

            # Il nome del file Sentinel-2 includerà la data dell'immagine e un indice sequenziale per l'incendio
            image_filename = f"fire_{fire_id}_{current_date_str}_pre_sentinel_{image_idx_in_fire}.tif"
            
            output_profile.update(count=len(band_arrays_resampled))
            
            out_path_tif = os.path.join(FIRE_SAVE_FOLDER, image_filename)
            with rasterio.open(out_path_tif, "w", **output_profile) as dst:
                dst.write(data_cube)
                dst.descriptions = band_names_resampled
            print(f"💾 Salvato immagine Sentinel (TIFF multi-banda): {out_path_tif}")

            # --- Generazione Ground Truth (GT) - UNA VOLTA PER INCENDIO ---
            if not gt_generated_for_fire:
                if gt_geometry_utm: 
                    if str(ref_src.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                        gt_geometry_for_rasterize = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(ref_src.crs).iloc[0]
                    else:
                        gt_geometry_for_rasterize = gt_geometry_utm

                    patch_bbox_for_gt_check = box(*rasterio.windows.bounds(window_to_read, ref_src.transform))
                    
                    if not gt_geometry_for_rasterize.intersects(patch_bbox_for_gt_check):
                         print("🔴 WARNING: La geometria GT proiettata NON INTERSECA la bounding box della patch di output. La maschera GT sarà vuota.")
                    
                    try:
                        gt_mask = rasterize(
                            shapes=[gt_geometry_for_rasterize], 
                            out_shape=(patch_size_pixels, patch_size_pixels),
                            transform=final_transform, 
                            fill=0, 
                            all_touched=True, 
                            default_value=1 
                        )
                        
                        if np.sum(gt_mask == 1) == 0:
                            print("🔴 WARNING: La maschera GT rasterizzata è completamente vuota (tutti 0). Possibili cause: geometria GT troppo piccola o non interseca la patch fissa.")
                        
                        gt_profile_tif = output_profile.copy() 
                        gt_profile_tif.update(
                            count=1,
                            dtype=rasterio.uint8, 
                            nodata=0 
                        )
                        out_path_gt_tif = os.path.join(FIRE_SAVE_FOLDER, f"fire_{fire_id}_GTSentinel.tif") 
                        with rasterio.open(out_path_gt_tif, "w", **gt_profile_tif) as dst:
                            dst.write(gt_mask.astype(rasterio.uint8), 1)
                        print(f"💾 Salvato maschera GT (TIFF) unica per incendio: {out_path_gt_tif}")

                        out_path_gt_png = os.path.join(FIRE_SAVE_FOLDER, f"fire_{fire_id}_GTSentinel.png")
                        save_gt_as_colored_png(gt_mask, out_path_gt_png)
                        print(f"💾 Salvato PNG colorato per GT unica: {out_path_gt_png}")
                        
                        gt_generated_for_fire = True 
                    except Exception as e:
                        print(f"❌ Errore durante la generazione della GT per {fire_id}: {e}")
                else:
                    print("⚠️ Geometria GT non valida. Maschera GT non generata.")

    print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Crea un config.yaml di esempio se non esiste (con i nuovi parametri)
    config_example = {
        "geojson_path": "data/shp_cp_2012_2024/piedmont_2012_2024_fa.geojson",
        "interval_pre_fire_days": 7, # AGGIORNATO
        "satellite": "sentinel-2",  # AGGIORNATO
        "patch_size_pixels": 256,
        "target_crs": "EPSG:32632", # NUOVO
        "min_fire_area_sq_m": 200,  # NUOVO
        "image_similarity_tolerance": 0.005 # NUOVO
    }
    
    os.makedirs("src/project_name", exist_ok=True)
    config_path = "src/project_name/config.yaml" # Definisci il percorso del config
    with open(config_path, "w") as f:
        yaml.dump(config_example, f)
    
    print(f"Generato o aggiornato il file di configurazione: {config_path}")


    config_data = load_config(config_path) # Carica il config dal percorso specificato

    print("\n--- Inizio Processo Batch per TUTTI gli incendi ---")

    gdf_all_fires = gpd.read_file(config_data["geojson_path"])
    min_date_for_sentinel2 = datetime(2015, 6, 23) 
    
    # Filtra gli incendi per data (solo quelli dopo l'inizio di Sentinel-2)
    gdf_all_fires_filtered = gdf_all_fires[pd.to_datetime(gdf_all_fires["initialdate"]) >= min_date_for_sentinel2]
    
    print(f"Trovati {len(gdf_all_fires)} incendi nel GeoJSON. Filtrati {len(gdf_all_fires_filtered)} incendi dopo il {min_date_for_sentinel2.strftime('%Y-%m-%d')}.")

    for idx, fire_row in gdf_all_fires_filtered.iterrows():
        try:
            process_single_fire(fire_row["id"], config_data)
        except Exception as e:
            print(f"🚨 ERRORE GRAVE durante il processo dell'incendio ID {fire_row['id']}: {e}")
            # Puoi aggiungere qui della logica per registrare gli errori o riprovare

    print("\n--- Processo Batch Completato ---")