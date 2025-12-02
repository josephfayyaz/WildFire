import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import box, shape
import pandas as pd
from datetime import datetime, timedelta

def process_single_landcover_for_fire(fire_id, fire_year, config):
    """
    Scarica il Landcover di LULC (10m) per un'area 256x256 pixel
    centrata sulla geometria dell'incendio e lo salva come file TIFF.
    Restituisce True in caso di successo, False altrimenti.
    """
    
    main_geojson_path = config["geojson_path"]
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    
    FIRE_SAVE_FOLDER = os.path.join(ROOT_DATASET_FOLDER, f"fire_{fire_id}")
    if not os.path.isdir(FIRE_SAVE_FOLDER):
        print(f"ℹ️ Cartella '{FIRE_SAVE_FOLDER}' non trovata. Nessuna azione intrapresa per incendio ID: {fire_id}.")
        return False 

    print(f"Caricamento GeoJSON da: {main_geojson_path}")
    try:
        gdf_all_fires = gpd.read_file(main_geojson_path)
    except Exception as e:
        print(f"❌ Errore nel caricamento del GeoJSON {main_geojson_path}: {e}")
        return False

    if "id" in gdf_all_fires.columns:
        if gdf_all_fires["id"].dtype != type(fire_id):
            print(f"DEBUG: Tipi di ID non corrispondenti: GeoJSON '{gdf_all_fires['id'].dtype}' vs Cercato '{type(fire_id)}'. Tentativo di conversione.")
            try:
                if gdf_all_fires["id"].dtype == object:
                    fire_data = gdf_all_fires[gdf_all_fires["id"] == str(fire_id)]
                else:
                    fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
            except Exception as e:
                print(f"Errore nella conversione o nel confronto dell'ID. Ricerco con tipo originale: {e}")
                fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
        else:
            fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
    else:
        print("DEBUG: AVVISO: Colonna 'id' non trovata nel GeoJSON. Assicurarsi che il GeoJSON abbia una colonna 'id'.")
        return False 

    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632") 
    if str(gdf_all_fires.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
        print(f"Convertendo GeoJSON da {gdf_all_fires.crs} a {TARGET_CRS_FOR_FIRES} per i calcoli interni.")
        gdf_all_fires = gdf_all_fires.to_crs(TARGET_CRS_FOR_FIRES)
        if "id" in gdf_all_fires.columns:
            if gdf_all_fires["id"].dtype == object:
                fire_data = gdf_all_fires[gdf_all_fires["id"] == str(fire_id)]
            else:
                fire_data = gdf_all_fires[gdf_all_fires["id"] == fire_id]
        else:
            print("AVVISO: Colonna 'id' non trovata nel GeoJSON dopo la conversione CRS.")
            return False


    if fire_data.empty:
        print(f"❌ Incendio con ID {fire_id} non trovato nel GeoJSON dopo la ricerca o la conversione CRS.")
        return False 

    fire = fire_data.iloc[0]
    gt_geometry_utm = shape(fire["geometry"])
    
    min_fire_area_sq_m = config.get("min_fire_area_sq_m", 200) 
    if gt_geometry_utm.area < min_fire_area_sq_m:
        print(f"❌ Incendio ID {fire_id} ha un'area troppo piccola ({gt_geometry_utm.area:.2f} mq) per essere utile a 10m di risoluzione. Minimo richiesto: {min_fire_area_sq_m} mq. Saltato.")
        print(f"\n--- FINE Processo per incendio ID: {fire_id} ---")
        return False

    TARGET_RESOLUTION_M = 10 
    patch_size_pixels = config.get("patch_size_pixels", 256)
    
    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M 

    print(f"\n--- Processando Land Cover per Incendio ID: {fire_id} ---")
    print(f"Dimensione patch target (Land Cover): {patch_size_pixels}x{patch_size_pixels} pixel ({fixed_patch_size_meters/1000:.2f} km per lato) a {TARGET_RESOLUTION_M}m.")
    print(f"Cartella di output: {FIRE_SAVE_FOLDER}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search_bbox_wgs84 = gpd.GeoSeries([gt_geometry_utm.buffer(fixed_patch_size_meters/1.5)], crs=TARGET_CRS_FOR_FIRES).to_crs(epsg=4326).iloc[0].bounds
    print(f"Ricerca STAC con bbox (WGS84): {search_bbox_wgs84}")
    
    stac_collections = ["io-lulc-annual-v02"] 
    
    # Estendi l'intervallo di ricerca per trovare più opzioni
    # Range da 5 anni prima a 5 anni dopo l'incendio, con limiti 2012-2024
    start_date_broad = datetime(max(2012, fire_year - 5), 1, 1) 
    end_date_broad = datetime(min(2024, fire_year + 5), 12, 31) 
    lulc_year_interval_broad = f"{start_date_broad.isoformat()}Z/{end_date_broad.isoformat()}Z"
    
    print(f"Ricerca STAC estesa per LULC nell'intervallo: {lulc_year_interval_broad}")

    search = catalog.search(
        collections=stac_collections,
        bbox=search_bbox_wgs84,
        limit=100, 
        datetime=lulc_year_interval_broad
    )

    items = list(search.items())
    
    if not items:
        print(f"❌ Nessun item Land Cover trovato per la regione {fire_id} nell'intervallo temporale esteso.")
        print(f"\n--- FINE Processo Land Cover per incendio ID: {fire_id} ---")
        return False

    print(f"✅ Trovati {len(items)} item Land Cover nell'intervallo esteso.")
    
    candidate_items_with_years = []
    
    for item_candidate in items:
        asset_name = "data" 
        if asset_name not in item_candidate.assets:
            continue
        
        signed_href_check = pc.sign(item_candidate.assets[asset_name]).href
        try:
            with rasterio.open(signed_href_check) as src_check:
                item_bounds_polygon = gpd.GeoSeries([box(*src_check.bounds)], crs=src_check.crs).iloc[0]
                
                if str(src_check.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                    gt_geometry_in_item_crs_check = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(src_check.crs).iloc[0]
                else:
                    gt_geometry_in_item_crs_check = gt_geometry_utm

                if gt_geometry_in_item_crs_check.intersects(item_bounds_polygon):
                    item_year = None
                    # Tentativo 1: Ottieni la data da 'datetime' property
                    item_datetime_str = item_candidate.properties.get('datetime')
                    if item_datetime_str:
                        try:
                            item_date_obj = datetime.fromisoformat(item_datetime_str.replace('Z', '+00:00'))
                            item_year = item_date_obj.year
                            print(f"  ✅ Item {item_candidate.id}: Anno ottenuto da 'datetime' property: {item_year}.")
                        except ValueError:
                            print(f"  ⚠️ Item {item_candidate.id}: 'datetime' property non è un formato data valido. Tentativo di estrazione anno dall'ID.")
                            item_year = None
                    
                    # Tentativo 2: Se 'datetime' non è valido o assente, estrai l'anno dall'ID
                    if item_year is None:
                        # Assumiamo un formato ID come '32T-2021'
                        parts = item_candidate.id.split('-')
                        if len(parts) > 1 and parts[-1].isdigit(): # Assicurati che l'ultima parte sia un numero
                            try:
                                item_year = int(parts[-1])
                                print(f"  ℹ️ Item {item_candidate.id}: Anno estratto da ID: {item_year}.")
                            except ValueError: 
                                print(f"  ⚠️ Item {item_candidate.id}: Impossibile convertire anno estratto da ID in numero. Ignorato.")
                                item_year = None
                        else:
                            print(f"  ⚠️ Item {item_candidate.id}: ID non nel formato 'XXX-YYYY' atteso per estrazione anno. Ignorato.")
                            item_year = None
                    
                    if item_year is not None:
                        candidate_items_with_years.append((item_year, item_candidate))
                    else:
                        print(f"  ⚠️ Item {item_candidate.id} interseca ma non è stato possibile ottenere una data valida. Ignorato per selezione data.")
                else:
                    print(f"  ❌ Item {item_candidate.id} NON interseca la geometria del fuoco.")
        except Exception as e:
            print(f"  Errore durante il controllo dell'item {item_candidate.id} per intersezione/data: {e}")
            
    if not candidate_items_with_years:
        print("🔴 Nessuna delle immagini Land Cover trovate interseca la geometria dell'incendio o ha metadati di tempo validi.")
        print(f"Impossibile procedere con il ritaglio per incendio ID: {fire_id}.")
        print(f"\n--- FINE Processo Land Cover per incendio ID: {fire_id} ---")
        return False

    # --- NUOVA LOGICA DI SELEZIONE: IL PIÙ VICINO ALL'ANNO DELL'INCENDIO ---
    
    # Calcola la differenza assoluta di anni per ogni candidato
    # E memorizza anche l'anno originale e l'item STAC
    items_with_diff = []
    for lc_year, item in candidate_items_with_years:
        diff = abs(lc_year - fire_year)
        items_with_diff.append((diff, lc_year, item))
    
    # Ordina: prima per differenza minima, poi per anno più recente (se la differenza è la stessa)
    # x[0] è la differenza, x[1] è l'anno. Usiamo -x[1] per ordinare in modo decrescente per l'anno (più recente)
    items_with_diff.sort(key=lambda x: (x[0], -x[1])) 
    
    selected_diff, selected_year, lc_item_to_process = items_with_diff[0]

    print(f"\n📸 Elaborazione Land Cover: {lc_item_to_process.id}.")
    print(f"Selezionato item LULC con anno {selected_year} (differenza {selected_diff} anni dall'incendio {fire_year}).")
    
    lc_asset_key = 'data' 
    signed_href = pc.sign(lc_item_to_process.assets[lc_asset_key]).href
    try:
        with rasterio.open(signed_href) as src_lc:
            print(f"DEBUG: Source Land Cover CRS: {src_lc.crs}, Bounds: {src_lc.bounds}, Resolution: {src_lc.res}, NoData Value: {src_lc.nodata}")
            
            fill_value = src_lc.nodata if src_lc.nodata is not None else 0 
            
            output_dtype = src_lc.profile['dtype'] if 'dtype' in src_lc.profile else np.uint8
            print(f"DEBUG: Determined output_dtype: {output_dtype}")

            if str(src_lc.crs).replace('epsg:', 'EPSG:') != TARGET_CRS_FOR_FIRES:
                gt_geometry_in_lc_crs = gpd.GeoSeries([gt_geometry_utm], crs=TARGET_CRS_FOR_FIRES).to_crs(src_lc.crs).iloc[0]
            else:
                gt_geometry_in_lc_crs = gt_geometry_utm
            
            centroid_x, centroid_y = gt_geometry_in_lc_crs.centroid.x, gt_geometry_in_lc_crs.centroid.y
            
            minx_patch_in_src_crs = centroid_x - (fixed_patch_size_meters / 2)
            miny_patch_in_src_crs = centroid_y - (fixed_patch_size_meters / 2)
            maxx_patch_in_src_crs = centroid_x + (fixed_patch_size_meters / 2)
            maxy_patch_in_src_crs = centroid_y + (fixed_patch_size_meters / 2)
            
            patch_bounds_in_src_crs = (minx_patch_in_src_crs, miny_patch_in_src_crs, maxx_patch_in_src_crs, maxy_patch_in_src_crs)

            window_to_read = from_bounds(*patch_bounds_in_src_crs, transform=src_lc.transform)

            window_to_read = window_to_read.intersection(rasterio.windows.Window(0, 0, src_lc.width, src_lc.height))
            window_to_read = window_to_read.round_offsets(op='floor').round_lengths(op='ceil')

            print(f"DEBUG: Calculated window to read from source Land Cover: {window_to_read}")
            src_window_bounds_in_src_crs = rasterio.windows.bounds(window_to_read, src_lc.transform)
            print(f"DEBUG: Bounds of calculated window (in source Land Cover CRS): {src_window_bounds_in_src_crs}")

            if window_to_read.width == 0 or window_to_read.height == 0:
                print("🔴 ERRORE: La finestra di lettura calcolata per il Land Cover è vuota o invalida. Incendio forse troppo vicino al bordo della tile Land Cover o dati mancanti. Skip.")
                print(f"\n--- FINE Processo Land Cover per incendio ID: {fire_id} ---")
                return False

            raw_window_data = src_lc.read(1, window=window_to_read, masked=True).squeeze()
            
            if isinstance(raw_window_data, np.ma.MaskedArray):
                lc_array_for_reproject = raw_window_data.filled(fill_value=fill_value)
            else:
                lc_array_for_reproject = raw_window_data 
            
            print(f"DEBUG: Dati raw dalla finestra di lettura - Shape (after squeeze and fill): {lc_array_for_reproject.shape}, Dtype: {lc_array_for_reproject.dtype}")
            print(f"DEBUG: Data for reproject - Min: {lc_array_for_reproject.min()}, Max: {lc_array_for_reproject.max()}")

            src_window_transform = rasterio.windows.transform(window_to_read, src_lc.transform)
            print(f"DEBUG: Transform of raw_window_data: {src_window_transform}")

            final_transform = rasterio.transform.from_bounds(
                gt_geometry_utm.centroid.x - (fixed_patch_size_meters / 2),
                gt_geometry_utm.centroid.y - (fixed_patch_size_meters / 2),
                gt_geometry_utm.centroid.x + (fixed_patch_size_meters / 2),
                gt_geometry_utm.centroid.y + (fixed_patch_size_meters / 2),
                width=patch_size_pixels, 
                height=patch_size_pixels
            )

            print(f"DEBUG: Final output transform: {final_transform}")
            print(f"DEBUG: Final output bounds (in target CRS {TARGET_CRS_FOR_FIRES}): {rasterio.transform.array_bounds(patch_size_pixels, patch_size_pixels, final_transform)}")

            lc_data_reprojected = np.zeros((patch_size_pixels, patch_size_pixels), dtype=output_dtype)

            reproject(
                source=lc_array_for_reproject, 
                destination=lc_data_reprojected,
                src_transform=src_window_transform, 
                src_crs=src_lc.crs, 
                dst_transform=final_transform, 
                dst_crs=TARGET_CRS_FOR_FIRES, 
                resampling=Resampling.nearest, 
                src_nodata=fill_value,
                dst_nodata=fill_value 
            )
            
            lc_filename = f"fire_{fire_id}_landcover.tif" 
            out_path_lc_tif = os.path.join(FIRE_SAVE_FOLDER, lc_filename) 

            output_profile = src_lc.profile.copy()
            output_profile.update({
                "height": patch_size_pixels,
                "width": patch_size_pixels,
                "count": 1,
                "dtype": lc_data_reprojected.dtype, 
                "crs": TARGET_CRS_FOR_FIRES,
                "transform": final_transform,
                "nodata": fill_value 
            })
            
            with rasterio.open(out_path_lc_tif, "w", **output_profile) as dst:
                dst.write(lc_data_reprojected, 1)
            print(f"💾 Salvato Land Cover (TIFF mono-banda a 10m): {out_path_lc_tif}")
            
            return True
    except Exception as e:
        print(f"❌ Errore critico durante la fase di download/elaborazione Land Cover per incendio ID {fire_id}: {e}")
        return False

def generate_landcover_for_specific_fire(single_fire_id: int, fire_year: int, config: dict):
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    fire_folder_path = os.path.join(ROOT_DATASET_FOLDER, f"fire_{single_fire_id}")

    print(f"\n--- Tentativo di generazione Land Cover per incendio ID: {single_fire_id} (Anno Incendio: {fire_year}) ---")
    print(f"Cercando la cartella: {fire_folder_path}")

    if os.path.isdir(fire_folder_path):
        print(f"✅ Cartella '{fire_folder_path}' trovata. Inizio generazione Land Cover.")
        success = process_single_landcover_for_fire(single_fire_id, fire_year, config)
        if success:
            print(f"Processo Land Cover completato con successo per incendio ID: {single_fire_id}.")
        else:
            print(f"Processo Land Cover fallito per incendio ID: {single_fire_id}. Controllare i log di errore.")
    else:
        print(f"ℹ️ Cartella '{fire_folder_path}' non trovata. Nessuna azione intrapresa per incendio ID: {single_fire_id}.")
    print(f"--- Fine tentativo per incendio ID: {single_fire_id} ---\n")


if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
    
    config = {
        "geojson_path": os.path.join(project_root_dir, "piedmont_geojson", "piedmont_2012_2024_fa.geojson"),
        "root_dataset_folder": os.path.join(project_root_dir, "piedmont_new"), 
        "target_crs": "EPSG:32632", 
        "min_fire_area_sq_m": 200,
        "patch_size_pixels": 256,
    }

    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERRORE: La cartella radice del dataset '{config['root_dataset_folder']}' non esiste.")
        print("Assicurati che 'root_dataset_folder' nella configurazione punti alla directory dove si trovano le cartelle degli incendi (es. 'piedmont_new').")
        exit("Impossibile procedere.")
    
    if not os.path.exists(config["geojson_path"]):
        print(f"ERRORE: Il file GeoJSON '{config['geojson_path']}' non esiste.")
        print("Assicurati che 'geojson_path' nella configurazione punti al file GeoJSON corretto.")
        exit("Impossibile procedere.")

    try:
        gdf_fires_metadata = gpd.read_file(config["geojson_path"])
        
        if 'id' not in gdf_fires_metadata.columns:
            print("ERRORE: La colonna 'id' non trovata nel GeoJSON. Necessaria per identificare gli incendi.")
            exit("Impossibile procedere.")
        if 'initialdate' not in gdf_fires_metadata.columns:
            print("ERRORE: La colonna 'initialdate' non trovata nel GeoJSON. Necessaria per ricavare l'anno dell'incendio.")
            exit("Impossibile procedere.")
        gdf_fires_metadata['start_year'] = pd.to_datetime(gdf_fires_metadata['initialdate']).dt.year
    except Exception as e:
        print(f"ERRORE: Impossibile caricare o processare il GeoJSON per ottenere gli anni degli incendi: {e}")
        exit("Impossibile procedere.")


    # --- MODIFICA QUI PER TESTARE UN SINGOLO ID ---
    # single_fire_id_to_test = 6100  # <--- INSERISCI QUI L'ID DELL'INCENDIO CHE VUOI TESTARE
    # 
    # fire_metadata_test = gdf_fires_metadata[gdf_fires_metadata['id'] == single_fire_id_to_test]
    # if not fire_metadata_test.empty:
    #     fire_year_for_test = int(fire_metadata_test.iloc[0]['start_year'])
    #     print(f"\n--- TEST SU SINGOLO INCENDIO ID: {single_fire_id_to_test} (Anno Incendio: {fire_year_for_test}) ---")
    #     generate_landcover_for_specific_fire(single_fire_id_to_test, fire_year_for_test, config)
    #     print(f"--- FINE TEST SU SINGOLO INCENDIO ID: {single_fire_id_to_test} ---\n")
    # else:
    #     print(f"Incendio ID {single_fire_id_to_test} non trovato nel GeoJSON. Impossibile procedere al test.")


    # Per runnare su tutti gli ID in futuro, decommenta il blocco seguente e commenta quello sopra:
    print(f'Avvio elaborazione dei raster Land Cover per tutti i folder in: {config["root_dataset_folder"]}')
    processed_count = 0
    for item in os.listdir(config["root_dataset_folder"]):
        full_path = os.path.join(config["root_dataset_folder"], item)
        
        if os.path.isdir(full_path) and item.startswith("fire_"):
            try:
                fire_id = int(item.replace("fire_", ""))
                
                fire_metadata = gdf_fires_metadata[gdf_fires_metadata['id'] == fire_id]
                if not fire_metadata.empty:
                    fire_year = int(fire_metadata.iloc[0]['start_year'])
                    success = process_single_landcover_for_fire(fire_id, fire_year, config)
                    if success:
                        processed_count += 1
                    else:
                        print(f"Processo Land Cover fallito per incendio ID: {fire_id}. (Anno: {fire_year})")
                else:
                    print(f"Warning: Dati GeoJSON per incendio ID {fire_id} non trovati. Saltato.")
            except ValueError:
                print(f"Warning: Il nome della cartella '{item}' non è nel formato 'fire_XXXX'. Saltato.")
            except Exception as e:
                print(f"Errore generale durante l'elaborazione della cartella '{item}': {e}")
    
    print(f"\nElaborazione completata. Generati raster Land Cover per {processed_count} cartelle di incendio.")
    print("\nScript terminato.")