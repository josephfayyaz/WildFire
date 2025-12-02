import os
import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from shapely.geometry import box, shape
from datetime import datetime, timedelta
import re
import time # Aggiunto import per time.sleep

# IMPORTANTE: Assicurati che questa importazione sia corretta per il tuo progetto
from modelWithLandsat.utils import find_best_image_in_folder

# --- Funzioni di Utilità ---

def parse_datetime_from_sentinel_filename(filepath: str) -> datetime | None:
    """
    Estrae la data di acquisizione da un nome di file Sentinel con il formato:
    'fire_xxxx_YYYY-MM-DD_pre_sentinel_N.tif' (dove xxxx è l'ID dell'incendio).

    Args:
        filepath (str): Il percorso completo o il nome del file Sentinel.

    Returns:
        datetime.datetime: L'oggetto datetime parsato (solo data), o None se il formato non corrisponde.
    """
    filename = os.path.basename(filepath)
    match = re.search(r'fire_\d+_(\d{4}-\d{2}-\d{2})_pre_sentinel_\d+\.tif', filename)
    
    if match:
        datetime_str = match.group(1)
        try:
            return datetime.strptime(datetime_str, '%Y-%m-%d')
        except ValueError:
            print(f"ATTENZIONE: Impossibile parsare la data '{datetime_str}' dal file: {filename}")
            return None
    else:
        print(f"ATTENZIONE: Nessun pattern di data trovato nel nome del file: {filename}")
        return None

# --- Funzioni per l'Estrazione di Bounding Box e Coordinate dal TIFF ---

def get_image_spatial_info(image_path: str, target_crs: str):
    """
    Estrae il centroide e la bounding box di un'immagine TIFF nel CRS target.

    Args:
        image_path (str): Percorso al file TIFF (es. Sentinel/Landsat).
        target_crs (str): Il CRS target (es. "EPSG:32632").

    Returns:
        tuple: (centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds) in WGS84 per la ricerca CDS.
               Restituisce (None, None, None) in caso di errore.
    """
    try:
        with rasterio.open(image_path) as src:
            src_bounds = src.bounds
            src_crs = src.crs

            # Calcola il centroide nel CRS sorgente
            centroid_x_src = (src_bounds.left + src_bounds.right) / 2
            centroid_y_src = (src_bounds.bottom + src_bounds.top) / 2
            
            # Crea un GeoDataFrame con il centroide nel CRS sorgente
            gdf_centroid_src = gpd.GeoDataFrame(
                geometry=[gpd.points_from_xy([centroid_x_src], [centroid_y_src])[0]],
                crs=src_crs
            )
            
            # Converte il centroide in WGS84 (EPSG:4326)
            gdf_centroid_wgs84 = gdf_centroid_src.to_crs(epsg=4326)
            centroid_lon_wgs84 = gdf_centroid_wgs84.geometry.x.iloc[0]
            centroid_lat_wgs84 = gdf_centroid_wgs84.geometry.y.iloc[0]
            
            # Calcola una bounding box in WGS84 basata sul centroide e una dimensione fissa.
            # Questo è per la richiesta CDS API. ERA5 ha una risoluzione di 0.1 gradi (~10km).
            # Una bbox di +/- 01 gradi (~50km) attorno al centroide dovrebbe coprire ampiamente
            # la patch di 2.56km x 2.56km e fornire dati sufficienti per l'interpolazione.
            # L'ordine per CDS API è [north, west, south, east]
            bbox_wgs84_for_cds = [
                centroid_lat_wgs84 + 0.1,  # north
                centroid_lon_wgs84 - 0.1,  # west
                centroid_lat_wgs84 - 0.1,  # south
                centroid_lon_wgs84 + 0.1   # east
            ]
            
            print(f"DEBUG: Immagine '{os.path.basename(image_path)}' - Centroide (WGS84): ({centroid_lat_wgs84:.4f}, {centroid_lon_wgs84:.4f})")
            print(f"DEBUG: Bounding box per richiesta CDS API (WGS84): {bbox_wgs84_for_cds}")
            
            return centroid_lon_wgs84, centroid_lat_wgs84, bbox_wgs84_for_cds
    except Exception as e:
        print(f"❌ Errore durante l'estrazione delle informazioni spaziali dall'immagine {image_path}: {e}")
        return None, None, None

# --- Funzione per il Download e Processamento Dati ERA5-Land dal CDS ---

def download_and_process_era5_land(fire_id: str, image_datetime: datetime, centroid_lon_wgs84: float, centroid_lat_wgs84: float, bbox_cds: list, config: dict) -> bool:
    """
    Scarica i dati ERA5-Land "derived-era5-land-daily-statistics" dal CDS API per la data e l'area specificate,
    li ritaglia e li riesampiona alla dimensione della patch desiderata (256x256 a 10m) in un unico TIFF multi-banda.
    """
    FIRE_SAVE_FOLDER = os.path.join(config["root_dataset_folder"], f"fire_{fire_id}")
    TARGET_RESOLUTION_M = config.get("target_resolution_m", 10) # Risoluzione target per i dati ERA5 (10m)
    patch_size_pixels = config.get("patch_size_pixels", 256)
    TARGET_CRS_FOR_FIRES = config.get("target_crs", "EPSG:32632")

    # Assicurati che la cartella di salvataggio esista
    os.makedirs(FIRE_SAVE_FOLDER, exist_ok=True)

    client = cdsapi.Client()
    dataset = "derived-era5-land-daily-statistics" # Dataset di statistiche giornaliere
    
    # Variabili ERA5-Land richieste con i loro nomi CDS e nomi Xarray corrispondenti.
    # L'ordine qui definirà l'ordine delle bande nel TIFF finale.
    era5_variables_config = [
        {"cds_name": "2m_dewpoint_temperature", "xr_name": "d2m"},
        {"cds_name": "2m_temperature", "xr_name": "t2m"},
        {"cds_name": "skin_temperature", "xr_name": "skt"},
        {"cds_name": "soil_temperature_level_1", "xr_name": "stl1"},
        {"cds_name": "soil_temperature_level_2", "xr_name": "stl2"},
        {"cds_name": "soil_temperature_level_3", "xr_name": "stl3"},
        {"cds_name": "soil_temperature_level_4", "xr_name": "stl4"},
        {"cds_name": "10m_u_component_of_wind", "xr_name": "u10"},
        {"cds_name": "10m_v_component_of_wind", "xr_name": "v10"},
        {"cds_name": "surface_pressure", "xr_name": "sp"},
    ]

    # Estrai tutti i nomi CDS delle variabili da richiedere
    all_cds_variables_to_request = [var["cds_name"] for var in era5_variables_config]
    
    # Nome del file temporaneo per il download del NetCDF contenente tutte le variabili
    temp_nc_file = os.path.join(FIRE_SAVE_FOLDER, f"temp_era5_land_all_vars_{image_datetime.strftime('%Y%m%d')}.nc")

    # Definisci la trasformazione finale per il reprojection (basata su centroide e dimensione patch)
    fixed_patch_size_meters = patch_size_pixels * TARGET_RESOLUTION_M # 256 * 10 = 2560 metri

    # Converti il centroide WGS84 al CRS target (es. UTM) per calcolare la trasformazione finale
    gdf_centroid_wgs84 = gpd.GeoDataFrame(
        geometry=[gpd.points_from_xy([centroid_lon_wgs84], [centroid_lat_wgs84])[0]],
        crs="EPSG:4326"
    )
    gdf_centroid_target_crs = gdf_centroid_wgs84.to_crs(TARGET_CRS_FOR_FIRES)
    centroid_x_target_crs = gdf_centroid_target_crs.geometry.x.iloc[0]
    centroid_y_target_crs = gdf_centroid_target_crs.geometry.y.iloc[0]

    # Calcola i bounds finali nel CRS target per la patch
    final_left = centroid_x_target_crs - (fixed_patch_size_meters / 2)
    final_right = centroid_x_target_crs + (fixed_patch_size_meters / 2)
    final_bottom = centroid_y_target_crs - (fixed_patch_size_meters / 2)
    final_top = centroid_y_target_crs + (fixed_patch_size_meters / 2)

    final_transform = rasterio.transform.from_bounds(
        final_left, final_bottom, final_right, final_top,
        width=patch_size_pixels,
        height=patch_size_pixels
    )
    final_output_crs = TARGET_CRS_FOR_FIRES

    reprojected_data_bands = [] # Lista per raccogliere gli array riproiettati di ogni banda
    band_names = [] # Per tenere traccia dei nomi delle bande

    ds_era5 = None # Inizializza ds_era5
    max_download_retries = 3
    download_successful_and_openable = False

    for attempt in range(max_download_retries):
        try:
            print(f"\n--- Scaricando tutte le variabili ERA5-Land per {image_datetime.strftime('%Y-%m-%d')} (Tentativo {attempt + 1}/{max_download_retries}) ---")
            request_params = {
                "variable": all_cds_variables_to_request, # Lista di tutte le variabili
                "year": str(image_datetime.year),
                "month": f"{image_datetime.month:02d}",
                "day": f"{image_datetime.day:02d}",
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "3_hourly",
                "area": bbox_cds,
                "format": "netcdf",
            }

            print(f"Richiesta CDS API per {len(all_cds_variables_to_request)} variabili...")
            client.retrieve(dataset, request_params, temp_nc_file).download()
            print(f"Download del file NetCDF combinato completato. Verificando...")
            file_size = os.path.getsize(temp_nc_file)
            if file_size == 0:
               raise ValueError(f"Il file scaricato è vuoto (0 bytes): {os.path.basename(temp_nc_file)}")
            print(f"Dimensione file scaricato: {file_size} bytes")

            # --- VERIFICA IMMEDIATA DEL FILE SCARICATO ---
            with xr.open_dataset(temp_nc_file, engine='netcdf4') as temp_ds:
                # Se arriviamo qui, il file è stato aperto con successo da xarray
                download_successful_and_openable = True
                print("Verifica del file NetCDF riuscita.")
                break # Esci dal ciclo di retry

        except Exception as e:
            print(f"❌ Errore durante il download o la verifica del file ERA5 combinato (Tentativo {attempt + 1}/{max_download_retries}) per incendio ID {fire_id}: {e}")
            if os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                print(f"File temporaneo {os.path.basename(temp_nc_file)} rimosso dopo errore.")
            
            if attempt < max_download_retries - 1:
                print(f"Ritentando download in 10 secondi...")
                time.sleep(10) # Aggiungi un ritardo prima di riprovare
            else:
                print(f"Tutti i {max_download_retries} tentativi di download sono falliti. Abortendo.")
                return False # Tutti i tentativi sono falliti, la funzione ritorna False

    # Se il download è avvenuto con successo e il file è apribile, procedi all'elaborazione
    if download_successful_and_openable:
        try:
            with xr.open_dataset(temp_nc_file, engine='netcdf4',chunks={}) as ds_era5: # Riapri per l'elaborazione effettiva
                success_all_vars_processing = True
                for var_info in era5_variables_config:
                    var_cds_name = var_info["cds_name"]
                    var_xr_name = var_info["xr_name"]
                    
                    print(f"Processando variabile: {var_cds_name} (internamente: {var_xr_name})")
                    
                    if var_xr_name not in ds_era5.variables:
                        print(f"❌ La variabile '{var_xr_name}' non è presente nel dataset NetCDF scaricato. Skip.")
                        success_all_vars_processing = False
                        continue

                    # Seleziona la variabile specifica per la data desiderata usando il nome corretto
                    era5_data_array_slice = ds_era5[var_xr_name].isel(valid_time=0).squeeze()

                    # Verifica che l'array sia 2D (lat, lon) dopo squeeze()
                    if len(era5_data_array_slice.dims) != 2:
                        print(f"❌ La selezione della variabile '{var_xr_name}' ha prodotto un array non 2D ({era5_data_array_slice.dims}). Skip.")
                        success_all_vars_processing = False
                        continue

                    era5_np_array = era5_data_array_slice.values # Converti in NumPy array
                    
                    # Ottieni la trasformazione del sorgente ERA5 per la riproiezione
                    min_lon_src = era5_data_array_slice.longitude.min().item()
                    max_lon_src = era5_data_array_slice.longitude.max().item()
                    min_lat_src = era5_data_array_slice.latitude.min().item()
                    max_lat_src = era5_data_array_slice.latitude.max().item()
                    
                    src_transform_era5 = rasterio.transform.from_bounds(
                        min_lon_src, min_lat_src, max_lon_src, max_lat_src,
                        era5_np_array.shape[1], era5_np_array.shape[0] # width, height (lon, lat)
                    )
                    src_crs_era5 = 'EPSG:4326' # CRS di ERA5 è WGS84 Lat/Lon

                    # Prepara l'array di destinazione vuoto per i dati riproiettati
                    era5_data_reprojected = np.zeros((patch_size_pixels, patch_size_pixels), dtype=np.float32)
                    
                    era5_fill_value = np.nan 
                    
                    reproject(
                        source=era5_np_array, 
                        destination=era5_data_reprojected,
                        src_transform=src_transform_era5, 
                        src_crs=src_crs_era5, 
                        dst_transform=final_transform, 
                        dst_crs=final_output_crs, 
                        resampling=Resampling.bilinear,
                        src_nodata=era5_fill_value,
                        num_threads=os.cpu_count()
                    )
                    
                    reprojected_data_bands.append(era5_data_reprojected)
                    band_names.append(var_cds_name)
                    print(f"Dati di '{var_cds_name}' preparati.")

            # Rimuovi il file temporaneo dopo aver processato tutte le variabili
            os.remove(temp_nc_file) 
            print(f"File temporaneo {os.path.basename(temp_nc_file)} rimosso.")

        except Exception as e:
            print(f"❌ Errore durante l'elaborazione del file NetCDF combinato per incendio ID {fire_id}: {e}")
            success_all_vars_processing = False
            # Se l'errore avviene qui, il file temporaneo potrebbe non essere stato rimosso.
            # Assicuriamoci che venga rimosso.
            if os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
                print(f"File temporaneo {os.path.basename(temp_nc_file)} rimosso dopo errore di processamento.")
            return False

    # Se tutte le variabili sono state processate con successo, crea il TIFF multi-banda
    if success_all_vars_processing and reprojected_data_bands:
        era5_output_filename = f"fire_{fire_id}_era5_multi_band_{image_datetime.strftime('%Y%m%d')}.tif"
        out_path_era5_tif = os.path.join(FIRE_SAVE_FOLDER, era5_output_filename)

        output_profile = {
            "height": patch_size_pixels,
            "width": patch_size_pixels,
            "count": len(reprojected_data_bands),
            "dtype": reprojected_data_bands[0].dtype,
            "crs": final_output_crs,
            "transform": final_transform,
            "nodata": -9999.0,
            "driver": "GTiff"
        }

        try:
            with rasterio.open(out_path_era5_tif, "w", **output_profile) as dst:
                for i, band_data in enumerate(reprojected_data_bands):
                    dst.write(band_data, i + 1)
                for i, name in enumerate(band_names):
                    dst.set_band_description(i + 1, name)

            print(f"🎉 Salvato TIFF multi-banda ERA5: {out_path_era5_tif}")
            print(f"Contiene le seguenti bande (nell'ordine): {', '.join(band_names)}")
            return True
        except Exception as e:
            print(f"❌ Errore durante la scrittura del TIFF multi-banda: {e}")
            return False
    else:
        print("⚠️ Nessun dato ERA5 riproiettato disponibile per la scrittura del TIFF multi-banda o errori precedenti.")
        return False


# --- Funzione Wrapper Principale ---

def generate_era5_for_specific_fire(single_fire_id: str, config: dict):
    """
    Funzione wrapper per avviare il processo di generazione ERA5 per un singolo incendio.

    Args:
        single_fire_id (str): L'ID dell'incendio da processare.
        config (dict): Dizionario di configurazione.
    """
    ROOT_DATASET_FOLDER = config["root_dataset_folder"]
    fire_folder_path = os.path.join(ROOT_DATASET_FOLDER, f"fire_{single_fire_id}")

    if not os.path.isdir(fire_folder_path):
        print(f"ℹ️ Cartella '{fire_folder_path}' non trovata. Nessuna azione intrapresa per incendio ID: {single_fire_id}.")
        return False

    print(f"\n--- Tentativo di generazione ERA5 per incendio ID: {single_fire_id} ---")
    print(f"Cercando la cartella: {fire_folder_path}")

    # 1. Ottieni il percorso dell'immagine Sentinel e la sua data
    best_image_info = find_best_image_in_folder(fire_folder_path)
    if best_image_info is None or 'sentinel_path' not in best_image_info:
        print(f"❌ Nessuna immagine Sentinel valida trovata per incendio ID: {single_fire_id}. Impossibile cercare ERA5.")
        return False
    
    pre_sentinel_image_path = best_image_info['sentinel_path']
    image_datetime = parse_datetime_from_sentinel_filename(pre_sentinel_image_path)

    if not image_datetime:
        print(f"❌ Impossibile determinare la data dell'immagine Sentinel dal percorso '{pre_sentinel_image_path}'. Necessaria per cercare ERA5.")
        return False
    
    print(f"Data immagine Sentinel trovata: {image_datetime.isoformat()} dal file: {os.path.basename(pre_sentinel_image_path)}")

    # 2. Estrai le coordinate e la bounding box dal file Sentinel
    centroid_lon_wgs84, centroid_lat_wgs84, bbox_cds = get_image_spatial_info(pre_sentinel_image_path, config["target_crs"])
    if centroid_lon_wgs84 is None:
        print(f"❌ Impossibile ottenere informazioni spaziali dall'immagine Sentinel. Impossibile procedere per incendio ID: {single_fire_id}.")
        return False

    # 3. Scarica e processa i dati ERA5-Land
    success = download_and_process_era5_land(fire_id=single_fire_id, 
                                            image_datetime=image_datetime, 
                                            centroid_lon_wgs84=centroid_lon_wgs84, 
                                            centroid_lat_wgs84=centroid_lat_wgs84, 
                                            bbox_cds=bbox_cds, 
                                            config=config) 
    
    if success:
        print(f"Processo ERA5 completato con successo per incendio ID: {single_fire_id}.")
    else:
        print(f"Processo ERA5 fallito per incendio ID: {single_fire_id}. Controllare i log di errore.")
    print(f"--- Fine tentativo per incendio ID: {single_fire_id} ---\n")


# --- Blocco Principale di Esecuzione (per test) ---

if __name__ == "__main__":
    # Configurazione del dataset e dei percorsi
    config = {
        "geojson_path": "piedmont_geojson/piedmont_2012_2024_fa.geojson", # Percorso al tuo GeoJSON degli incendi
        "root_dataset_folder": "piedmont_new", # La cartella principale dove sono salvati gli incendi (es. 'piedmont_new/fire_ID/')
        "target_crs": "EPSG:32632", # CRS per l'area del Piemonte (UTM zona 32N)
        "patch_size_pixels": 256, # Dimensione della patch in pixel (es. 256x256)
        "target_resolution_m": 10, # Risoluzione finale dei dati ERA5 in metri (come richiesto)
    }

    # ID dell'incendio da processare per test.
    # Assicurati che esista una cartella come 'piedmont_new/fire_6500'
    # e che contenga un file Sentinel .tif che segue il pattern 'fire_xxxx_YYYY-MM-DD_pre_sentinel_N.tif'
    fire_id_to_process = "6500" # Sostituisci con l'ID incendio desiderato (stringa)

    # Verifica che la cartella radice del dataset esista prima di procedere
    if not os.path.exists(config["root_dataset_folder"]):
        print(f"ERRORE: La cartella radice del dataset '{config['root_dataset_folder']}' non esiste.")
        print("Assicurati che 'root_dataset_folder' nella configurazione punti alla directory dove si trovano le cartelle degli incendi (es. 'piedmont_new').")
        exit("Impossibile procedere.")
    
    # Esempio di utilizzo per un singolo incendio
    generate_era5_for_specific_fire(fire_id_to_process, config)

    # Puoi aggiungere altre chiamate a 'generate_era5_for_specific_fire' qui
    # per processare altri incendi, ad esempio:
    # generate_era5_for_specific_fire("ALTRO_ID_INCENDIO", config)
    # generate_era5_for_specific_fire("ANCORA_ALTRO_ID_INCENDIO", config)