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

# Funzioni di supporto (possono stare in un utility.py o all'inizio del main script)
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
    # print(f"💾 Salvato PNG colorato per GT: {png_path}") # Rimosso per meno verbosità

# Funzione per confrontare due immagini (array NumPy)
def are_images_similar(img1, img2, tolerance_percentage=0.01):
    """
    Compara due array NumPy (immagini) per verificarne la somiglianza.
    Ritorna True se la differenza relativa media per pixel è inferiore alla tolleranza.
    img1 e img2 devono avere la stessa shape e dtype.
    """
    if img1.shape != img2.shape or img1.dtype != img2.dtype:
        return False # Formati diversi non sono simili

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    max_val = np.iinfo(img1.dtype).max if np.issubdtype(img1.dtype, np.integer) else 1.0 # Max valore del tipo di dati
    
    # Calcola la differenza relativa media
    # Usiamo mean_absolute_percentage_error (MAPE) come proxy, ma normalizzato
    mean_relative_diff = np.mean(diff / max_val)
    
    return mean_relative_diff < tolerance_percentage


def process_single_fire(fire_id, config):
    """
    Processa un singolo incendio, cercando *tutte* le immagini Landsat-2 valide,
    ritagliandole e generando la maschera GT con una dimensione geografica fissa.
    Aggiunto controllo di somiglianza per immagini dello stesso giorno.
    """
    
    main_geojson_path = config["geojson_path"]
    SAVE_ROOT_FOLDER = "dataset_output" 
    
    os.makedirs(SAVE_ROOT_FOLDER, exist_ok=True)
    
    print(f"Caricamento GeoJSON da: {main_geojson_path}")
    gdf_all_fires = gpd.read_file(main_geojson_path)
    
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
    
    min_fire_area_sq_m = config.get("min_fire_area_sq_m", 200) 
    if gt_geometry_utm.area < min_fire_area_sq_m:
        print(f"❌ Incendio ID {fire_id} ha un'area troppo piccola ({gt_geometry_utm.area:.2f} mq) per essere utile a 10m di risoluzione. Minimo richiesto: {min_fire_area_sq_m} mq. Saltato.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    TARGET_RESOLUTION_M = 30 
    patch_size_pixels = config.get("patch_size_pixels", 256)
    interval_days = config.get("interval", 7) 

    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M 

    FIRE_SAVE_FOLDER = os.path.join(SAVE_ROOT_FOLDER, f"fire_{fire_id}")
    os.makedirs(FIRE_SAVE_FOLDER, exist_ok=True)

    buffer_for_stac_search_meters = fixed_patch_size_meters / 2 
    fire_geometry_search_buffer_utm = gt_geometry_utm.buffer(buffer_for_stac_search_meters)
    fire_geometry_search_buffer_wgs84 = gpd.GeoSeries([fire_geometry_search_buffer_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(epsg=4326).iloc[0]

    time_range_start = fire_date
    time_range_end = fire_date + timedelta(days=interval_days)
    time_range_str = f"{time_range_start.strftime('%Y-%m-%d')}/{time_range_end.strftime('%Y-%m-%d')}"

    print(f"\n--- Processando Incendio ID: {fire_id} ---")
    print("Data dell'incendio:", fire_date.strftime('%Y-%m-%d'))
    print("Intervallo di ricerca STAC:", time_range_str)
    print(f"Dimensione patch target (finale): {patch_size_pixels}x{patch_size_pixels} pixel ({fixed_patch_size_meters/1000:.2f} km per lato).")
    print(f"Cartella di output: {FIRE_SAVE_FOLDER}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    
    print(f"Ricerca STAC con bbox (WGS84): {fire_geometry_search_buffer_wgs84.bounds}")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=fire_geometry_search_buffer_wgs84.bounds,
        datetime=time_range_str,
        limit=500 
    )

    items = list(search.items())
    if not items:
        print("❌ Nessuna immagine Landsat-2 trovata per la regione/data.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    print(f"✅ Trovate {len(items)} immagini Landsat-2.")

    landsat_band_resolutions = {
        "blue": 30, "green": 30, "red": 30, "nir08": 30, "swir16": 30, "swir22": 30,
        "lwir11": 30, "lwir": 30, "coastal": 30,
        "atran": 30, "cdist": 30, "drad": 30, "emis": 30, "emsd": 30, "trad": 30, "urad": 30
    }
    
    valid_items = []
    
    for i, item_candidate in enumerate(items):
        ref_band_name = "red"
        if ref_band_name not in item_candidate.assets:
             ref_band_name = next((b for b, r in landsat_band_resolutions.items() if r == 10 and b in item_candidate.assets), None)
        
        if ref_band_name is None:
            print(f"  Saltata immagine {i+1} del {item_candidate.datetime.strftime('%Y-%m-%d')} - Nessuna banda a 30m disponibile.")
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
        print("🔴 Nessuna delle immagini Landsat-2 trovate interseca la geometria dell'incendio o ha bande valide. Impossibile procedere con il ritaglio.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return

    print(f"\n✅ Iniziamo l'elaborazione di {len(valid_items)} immagini valide.")

    # Dizionario per memorizzare le immagini già processate per la stessa data, per il confronto
    processed_images_by_date = {} 
    
    # Contatore per gestire immagini con la stessa data di acquisizione (per il nome del file)
    date_counts = {}

    # Loop attraverso TUTTI i valid_items
    for item_idx, item in enumerate(valid_items):
        current_date_str = item.datetime.strftime('%Y-%m-%d')
        
        print(f"\n📸 Elaborazione immagine {item_idx + 1}/{len(valid_items)}: {item.id} del {current_date_str}.")
        
        band_arrays_resampled = []
        band_names_resampled = []
        
        reference_band_name = "red"
        if reference_band_name not in item.assets:
             reference_band_name = next((b for b, r in landsat_band_resolutions.items() if r == 10 and b in item.assets), None)

        if reference_band_name is None:
            print("⚠️ Nessuna banda a 30m trovata per il riferimento nell'immagine corrente. Skip questa immagine.")
            continue
        
        signed_href = planetary_computer.sign(item.assets[reference_band_name]).href
        with rasterio.open(signed_href) as ref_src:
            if str(ref_src.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES: 
                 print(f"🔴 CRITICAL WARNING: Il CRS del GeoJSON ({TARGET_CRS_FOR_FIRES}) NON corrisponde al CRS dell'immagine Landsat ({ref_src.crs})! Questo causerà disallineamenti.")
            
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
                print("🔴 ERRORE: La finestra di lettura calcolata è vuota o invalida dopo l'intersezione con i limiti dell'immagine Landsat. Incendio forse troppo vicino al bordo della tile o dati mancanti. Skip questa immagine.")
                continue

            print(f"Window di lettura calcolata per la patch fissa (nell'immagine Landsat originale): {window_to_read}")
            print(f"Dimensione (pixel) della window di lettura: ({int(window_to_read.width)}, {int(window_to_read.height)})")
            
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
                "count": len(landsat_band_resolutions),
                "dtype": np.uint16 
            })

            # Lista per memorizzare le bande a 30m per il confronto di somiglianza
            bands_30m_for_comparison = [] 

            for band, native_resolution in landsat_band_resolutions.items():
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

                        if native_resolution == 10: # Aggiungi solo le bande a 30m per il confronto
                            bands_30m_for_comparison.append(band_data)

                    except Exception as e:
                        print(f"  ❌ Errore su banda {band} durante la lettura/resampling: {e}")

            if not band_arrays_resampled:
                print("⚠️ Nessuna banda utile trovata per questa patch. Impossibile salvare immagine Landsat.")
                continue 

            data_cube = np.stack(band_arrays_resampled, axis=0)

            # --- Controllo di Somiglianza ---
            if current_date_str in processed_images_by_date:
                # Confronta le bande a 10m dell'immagine corrente con quelle già elaborate per la stessa data
                current_30m_bands_stack = np.stack(bands_30m_for_comparison, axis=0) if bands_30m_for_comparison else None
                
                is_similar_to_existing = False
                for existing_30m_bands_stack in processed_images_by_date[current_date_str]:
                    if current_30m_bands_stack is not None and are_images_similar(current_30m_bands_stack, existing_30m_bands_stack, tolerance_percentage=0.005): # Tolleranza regolabile (0.5%)
                        is_similar_to_existing = True
                        break
                
                if is_similar_to_existing:
                    print(f"ℹ️ Immagine del {current_date_str} è troppo simile a una già processata. Saltata per evitare duplicati.")
                    continue
                else:
                    # Aggiungi l'immagine corrente al registro delle immagini processate per questa data
                    processed_images_by_date[current_date_str].append(current_30m_bands_stack)
            else:
                # Prima immagine per questa data, aggiungila al registro
                if bands_30m_for_comparison:
                    processed_images_by_date[current_date_str] = [np.stack(bands_30m_for_comparison, axis=0)]
                else:
                    processed_images_by_date[current_date_str] = [] # Non ha bande a 10m per confronto, ma può comunque essere processata

            # --- Fine Controllo Somiglianza ---

            output_profile.update(count=len(band_arrays_resampled))
            
            # Aggiorna il contatore data_counts SOLO se l'immagine è stata approvata dal controllo somiglianza
            date_counts[current_date_str] = date_counts.get(current_date_str, 0) + 1
            full_date_str = f"{current_date_str}_{date_counts[current_date_str]}" if date_counts[current_date_str] > 1 else current_date_str

            out_path_tif = os.path.join(FIRE_SAVE_FOLDER, f"fire_{fire_id}_{full_date_str}_Landsat_patch.tif")
            with rasterio.open(out_path_tif, "w", **output_profile) as dst:
                dst.write(data_cube)
                dst.descriptions = band_names_resampled
            print(f"💾 Salvato immagine Landsat (TIFF multi-banda): {out_path_tif}")

            # --- Generazione Ground Truth (GT) ---
            # La GT è generata per ogni immagine perché la `final_transform` è specifica di ogni immagine
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
                    out_path_gt_tif = os.path.join(FIRE_SAVE_FOLDER, f"fire_{fire_id}_{full_date_str}_GT_mask.tif") 
                    with rasterio.open(out_path_gt_tif, "w", **gt_profile_tif) as dst:
                        dst.write(gt_mask.astype(rasterio.uint8), 1)
                    print(f"💾 Salvato maschera GT (TIFF): {out_path_gt_tif}")

                    out_path_gt_png = os.path.join(FIRE_SAVE_FOLDER, f"fire_{fire_id}_{full_date_str}_GT_mask.png")
                    save_gt_as_colored_png(gt_mask, out_path_gt_png)

                except Exception as e:
                    print(f"❌ Errore durante la generazione della GT per {fire_id} e data {full_date_str}: {e}")
            else:
                print("⚠️ Geometria GT non valida. Maschera GT non generata per questa immagine.")

    print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    config_data = load_config()

    # Per testare un singolo incendio che sai avere più immagini (e magari simili nello stesso giorno)
    fire_id_to_test = 5440 # Sostituisci con un ID di test
    process_single_fire(fire_id_to_test, config_data)

    # Per processare tutti gli incendi
    # gdf_all_fires = gpd.read_file(config_data["geojson_path"])
    # min_date_for_Landsat2 = datetime(2015, 6, 23) 
    # gdf_all_fires_filtered = gdf_all_fires[pd.to_datetime(gdf_all_fires["initialdate"]) >= min_date_for_Landsat2]
    # 
    # for idx, fire_row in gdf_all_fires_filtered.iterrows():
    #     process_single_fire(fire_row["id"], config_data)