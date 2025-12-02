import os
import torch
import rasterio
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List 
import cv2
import albumentations as A
from augmentations import SentinelAugmentations 
from albumentations.pytorch import ToTensorV2
import geopandas as gpd
import pandas as pd
from datetime import datetime
import re
from utils import find_best_image_in_folder 


# Costanti per il numero di bande, utili per chiarezza e per evitare numeri magici
N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15 # Numero di bande Landsat "reali" (senza la banda di flag)
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1 # 15 + 1 = 16
N_BANDS_ERA5 = 3
N_BANDS_ERA5_RASTER = 2 # Numero di bande raster (es. vento u/v)
N_BANDS_ERA5_TABULAR = 1 # Numero di bande tabulari 
ERA5_TABULAR_BAND_INDICES = [0] # Indici delle bande tabulari all'interno dell'array ERA5 completo
ERA5_RASTER_BAND_INDICES = [1, 2]
LANDCOVER_CLASSES = 12


# Gli indici REALI delle bande ERA5 nel tuo array di ingresso.

class PiedmontDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, geojson_path: str,
                 target_size: Tuple[int, int] = (256, 256),
                 compute_stats: bool = True,
                 apply_augmentations: bool = False,
                 global_stats: Optional[Dict[str, torch.Tensor]] = None, 
                 initial_fire_dirs: Optional[List[str]] = None 
                 ): 
        
        self.root_dir = root_dir
        self.geojson_path = geojson_path
        self.target_size = target_size
        self.apply_augmentations = apply_augmentations

        self.fire_id_to_date_map = self._load_fire_dates_from_geojson()
        
        # --- MODIFICA 1: Aggiunto un attributo per memorizzare la disponibilità di Landsat per ogni campione. ---
        self.sample_type_map: Dict[str, str] = {} 

        ##MOD:
        
        if initial_fire_dirs is not None:
            self.fire_dirs = initial_fire_dirs
            print(f"Dataset initialized with {len(self.fire_dirs)} pre-filtered directories.")
            self._populate_sample_type_map_for_initial_dirs()
        else:
            ###For the first initialization, calculating global stats
            print("Filtering valid directories from root_dir...")
            # --- MODIFICA 2: _filter_valid_directories ora popola self.sample_type_map ---
            self.fire_dirs = self._filter_valid_directories() 
            print(f"Found {len(self.fire_dirs)} valid directories.")
        
        # Gestione delle statistiche globali: calcola solo se richiesto E non fornite
        if compute_stats and global_stats is None:
            # --- MODIFICA 3: _compute_global_stats_mean_std ora usa self.sample_type_map ---
            # E CERCA IL LANDSAT 10M PER LE STATS
            self.global_stats = self._compute_global_stats_mean_std()
        elif global_stats is not None:
            self.global_stats = global_stats
        else:
            self.global_stats = None
            
        self.bin_thresholds = [1000, 5000, 20000] # Soglie in pixel
        self.sample_pixel_counts = {}
        self.sample_bins = {}
        self._map_samples_to_bins()


        ##Train
        self.augmentor = SentinelAugmentations(
            target_size=self.target_size
        )

        ##Val
        self.eval_transform = A.Compose([
            ToTensorV2()
        ])
    
    def _populate_sample_type_map_for_initial_dirs(self):
        """
        Popola self.sample_type_map per le directory già caricate,
        usato quando initial_fire_dirs non è None.
        """
        print("Populating sample_type_map for pre-filtered directories...")
        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)
            has_sentinel = any("pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f for f in files)
            # QUI: Cerca il Landsat 10m per determinare la presenza
            has_landsat = any("pre_landsat" in f and "_10m.tif" in f and "_CM" not in f for f in files)
            
            if has_sentinel and has_landsat:
                self.sample_type_map[fire_dir] = 'both'
            elif has_sentinel:
                self.sample_type_map[fire_dir] = 'sentinel_only'
            else:
                self.sample_type_map[fire_dir] = 'unknown' 
        print(f"Populated sample_type_map with {len(self.sample_type_map)} entries.")

    def _load_fire_dates_from_geojson(self) -> Dict[int, datetime]:
        # ... (il tuo codice esistente, nessuna modifica qui) ...
        fire_id_to_date = {}
        try:
            gdf = gpd.read_file(self.geojson_path)

            if 'id' not in gdf.columns or 'initialdate' not in gdf.columns:
                raise ValueError("GeoJSON file must contain 'id' and 'initialdate' columns.")

            gdf['parsed_date'] = pd.to_datetime(gdf['initialdate'], errors='coerce')

            for index, row in gdf.iterrows():
                fire_id = row['id']
                fire_date_timestamp = row['parsed_date']

                if fire_id is not None and pd.notna(fire_date_timestamp):
                    fire_id_to_date[int(fire_id)] = fire_date_timestamp.to_pydatetime()
                else:
                    print(f"Warning: Skipping GeoJSON feature with ID '{fire_id}' due to missing or unparseable initialdate: '{row['initialdate']}'.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: GeoJSON file not found at {self.geojson_path}. Please check the path.")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing GeoJSON file {self.geojson_path}: {e}")

        print(f"Loaded {len(fire_id_to_date)} fire dates from GeoJSON using geopandas.")
        return fire_id_to_date

    def _filter_valid_directories(self) -> list:
        """
        Filters directories to include only those that:
        1. Contain at least one 'pre_sentinel' file.
        2. Contain a 'GTSentinel' ground truth file.
        
        Additionally, it populates `self.sample_type_map` to indicate if Landsat (10m resampled) is also present.
        This allows the dataset to provide 'sentinel_only' samples if Landsat is missing.
        """
        valid_dirs = []
        all_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir)
                                if os.path.isdir(os.path.join(self.root_dir, d))]

        # Contatori per il debug
        count_sentinel_only = 0
        count_both = 0
        count_missing_gt = 0
        count_no_sentinel = 0
        
        print(f"Scanning {len(all_dirs)} directories in '{self.root_dir}'...")

        for fire_dir in all_dirs:
            dir_name = os.path.basename(fire_dir)

            match = re.search(r'fire_(\d+)', dir_name)
            fire_id = int(match.group(1)) if match else None

            if fire_id is None or fire_id not in self.fire_id_to_date_map:
                continue

            files = os.listdir(fire_dir)
            
            # 1. Check for 'pre_sentinel' file
            has_sentinel = any("pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f for f in files)
            
            # 2. Check for 'pre_landsat_10m' file
            # QUI: Verifichiamo l'esistenza del file Landsat già risampled a 10m
            has_landsat = any("pre_landsat" in f and "_10m.tif" in f and "_CM" not in f for f in files)
            
            # 3. Check for 'GTSentinel' ground truth file
            has_gt_sentinel = any("GTSentinel" in f and f.endswith(".tif") for f in files)

            # Il criterio di validità ora richiede SOLO Sentinel e GT. Landsat è opzionale.
            if has_sentinel and has_gt_sentinel:
                valid_dirs.append(fire_dir)
                if has_landsat:
                    self.sample_type_map[fire_dir] = 'both'
                    count_both += 1
                else:
                    self.sample_type_map[fire_dir] = 'sentinel_only'
                    count_sentinel_only += 1
            else:
                if not has_sentinel:
                    count_no_sentinel += 1
                if not has_gt_sentinel:
                    count_missing_gt += 1
                continue # Non aggiunge alla lista dei valid_dirs

        print(f"Filtering complete.")
        print(f"   - Total valid directories (with Sentinel and GT): {len(valid_dirs)}")
        print(f"   - Of these, {count_both} have both Sentinel and Landsat (10m) data.")
        print(f"   - Of these, {count_sentinel_only} have only Sentinel data (Landsat is missing/placeholder).")
        print(f"   - Skipped {count_no_sentinel} directories due to missing Sentinel data.")
        print(f"   - Skipped {count_missing_gt} directories due to missing GTSentinel mask.") # Potrebbe sovrapporsi con no_sentinel
        
        return valid_dirs

    def _compute_global_stats_mean_std(self) -> Dict[str, torch.Tensor]:
        """
        Computes global mean and standard deviation for each of the 12 Sentinel-2 bands,
        15 Landsat bands, DEM, and split ERA5 (raster and tabular components).
        Iterates through all *first* available images in the dataset.
        For Landsat, only samples where Landsat data is actually present (10m resampled) are used for stats.
        """
        print("Computing global mean and standard deviation for normalization...")

        sum_values_sentinel = torch.zeros(N_BANDS_SENTINEL, dtype=torch.float64)
        sum_sq_values_sentinel = torch.zeros(N_BANDS_SENTINEL, dtype=torch.float64)
        count_values_sentinel = torch.zeros(N_BANDS_SENTINEL, dtype=torch.int64)

        sum_values_landsat = torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float64)
        sum_sq_values_landsat = torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float64)
        count_values_landsat = torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.int64)

        sum_values_era5_raster = torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float64)
        sum_sq_values_era5_raster = torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float64)
        count_values_era5_raster = torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.int64)
        
        sum_values_era5_tabular = torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float64)
        sum_sq_values_era5_tabular = torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float64)

        sum_values_dem = torch.zeros(1, dtype=torch.float64)
        sum_sq_values_dem = torch.zeros(1, dtype=torch.float64)
        count_values_dem = torch.zeros(1, dtype=torch.int64)

        num_sentinel_samples_for_stats = 0
        num_landsat_samples_for_stats = 0
        num_dem_samples_for_stats = 0
        num_era5_samples_for_stats = 0 # Questo conterà quanti campioni ERA5 validi abbiamo processato

        for fire_dir in self.fire_dirs:
            files = os.listdir(fire_dir)

            # DEM stats (NESSUNA MODIFICA)
            dem_paths_in_dir = [f for f in files if "dem_10m" in f and f.endswith(".tif")]
            if dem_paths_in_dir: # Aggiunto controllo per lista non vuota per evitare IndexError
                dem_path_file = dem_paths_in_dir[0] # Usa un nome diverso per evitare confusione
                path_dem = os.path.join(fire_dir, dem_path_file)
                try:
                    img_np_dem = self._read_image(path_dem)
                    img_np_dem = np.nan_to_num(img_np_dem, nan=0.0, posinf=0.0, neginf=0.0)
                    img_np_dem = np.clip(img_np_dem, 0.0, 10000.0).astype(np.float64)

                    band_data = img_np_dem[:, :, 0].flatten() # Solo una banda per DEM
                    if band_data.size > 0:
                        sum_values_dem += np.sum(band_data)
                        sum_sq_values_dem += np.sum(band_data**2)
                        count_values_dem += len(band_data)
                    else:
                        print(f"[DEBUG STATS] Warning: Band in {path_dem} has no valid data after clipping/nan_to_num for stats.")
                    num_dem_samples_for_stats += 1
                except Exception as e:
                    print(f"     [STATS ERROR] Error processing {path_dem} for DEM stats: {e}")

            # ERA 5 Stats (LE UNICHE MODIFICHE SIGNIFICATIVE QUI)
            era5_paths_in_dir = [f for f in files if "era5" in f and f.endswith(".tif")]
            if era5_paths_in_dir: # Aggiunto controllo per lista non vuota
                era5_path_file = era5_paths_in_dir[0] # Usa un nome diverso
                path_era5 = os.path.join(fire_dir, era5_path_file)
                try:
                    img_np_era5 = self._read_image(path_era5)
                    img_np_era5 = np.nan_to_num(img_np_era5, nan=0.0, posinf=0.0, neginf=0.0)
                    img_np_era5 = np.clip(img_np_era5, 0.0, 10000.0).astype(np.float64)

                    if img_np_era5.shape[2] != N_BANDS_ERA5:
                        print(f"     [DEBUG STATS] Skipping {path_era5}: expected {N_BANDS_ERA5} bands, got {img_np_era5.shape[2]}.")
                        continue

                    # Processa le bande ERA5 Raster (es. vento u, v) - pixel-wise
                    for i, original_band_idx in enumerate(ERA5_RASTER_BAND_INDICES):
                        band_data = img_np_era5[:, :, original_band_idx].flatten()
                        if band_data.size > 0:
                            sum_values_era5_raster[i] += np.sum(band_data)
                            sum_sq_values_era5_raster[i] += np.sum(band_data**2)
                            count_values_era5_raster[i] += len(band_data)
                        else:
                            print(f"[DEBUG STATS] Warning: ERA5 Raster Band (original idx {original_band_idx}) in {path_era5} has no valid pixel data for stats.")
                    
                    # Processa le bande ERA5 Tabulari (es. temperatura, pressione) - sample-wise (media spaziale)
                    for i, original_band_idx in enumerate(ERA5_TABULAR_BAND_INDICES):
                        band_spatial_mean = np.mean(img_np_era5[:, :, original_band_idx])
                        sum_values_era5_tabular[i] += band_spatial_mean
                        sum_sq_values_era5_tabular[i] += band_spatial_mean**2


                    num_era5_samples_for_stats += 1 # Conta i campioni validi che hanno contribuito
                except Exception as e:
                    print(f"     [STATS ERROR] Error processing {path_era5} for ERA5 stats: {e}")

            # Sentinel Stats (NESSUNA MODIFICA)
            pre_sentinel_files_in_dir = sorted([
                f for f in files
                if "pre_sentinel" in f and f.endswith(".tif") and "_CM" not in f
            ])
            first_pre_sentinel_file = pre_sentinel_files_in_dir[0] if pre_sentinel_files_in_dir else None

            if first_pre_sentinel_file:
                path_sentinel = os.path.join(fire_dir, first_pre_sentinel_file)
                try:
                    img_np_sentinel = self._read_image(path_sentinel)
                    img_np_sentinel = np.nan_to_num(img_np_sentinel, nan=0.0, posinf=0.0, neginf=0.0)
                    img_np_sentinel = np.clip(img_np_sentinel, 0.0, 10000.0).astype(np.float64)

                    if img_np_sentinel.shape[2] != N_BANDS_SENTINEL:
                        print(f"     [DEBUG STATS] Skipping {path_sentinel}: expected {N_BANDS_SENTINEL} bands, got {img_np_sentinel.shape[2]}.")
                        continue

                    for band_idx in range(N_BANDS_SENTINEL):
                        band_data = img_np_sentinel[:, :, band_idx].flatten()
                        if band_data.size > 0:
                            sum_values_sentinel[band_idx] += np.sum(band_data)
                            sum_sq_values_sentinel[band_idx] += np.sum(band_data**2)
                            count_values_sentinel[band_idx] += len(band_data)
                        else:
                            print(f"[DEBUG STATS] Warning: Band {band_idx} in {path_sentinel} has no valid data after clipping/nan_to_num for stats.")
                    num_sentinel_samples_for_stats += 1
                except Exception as e:
                    print(f"     [STATS ERROR] Error processing {path_sentinel} for Sentinel stats: {e}")

            # Landsat stats (only if available) (NESSUNA MODIFICA)
            sample_type = self.sample_type_map.get(fire_dir, 'unknown')
            if sample_type == 'both':
                pre_landsat_files_in_dir = sorted([
                    f for f in files
                    if "pre_landsat" in f and "_10m.tif" in f and "_CM" not in f
                ])
                first_pre_landsat_file = pre_landsat_files_in_dir[0] if pre_landsat_files_in_dir else None

                if first_pre_landsat_file:
                    path_landsat = os.path.join(fire_dir, first_pre_landsat_file)
                    try:
                        img_np_landsat = self._read_image(path_landsat)
                        img_np_landsat = np.nan_to_num(img_np_landsat, nan=0.0, posinf=0.0, neginf=0.0)

                        if img_np_landsat.shape[2] != N_BANDS_LANDSAT_ACTUAL:
                            print(f"     [DEBUG STATS] Skipping {path_landsat}: expected {N_BANDS_LANDSAT_ACTUAL} bands, got {img_np_landsat.shape[2]}.")
                            continue

                        for band_idx in range(N_BANDS_LANDSAT_ACTUAL): # Only real bands
                            band_data = img_np_landsat[:, :, band_idx].flatten()
                            if band_data.size > 0:
                                sum_values_landsat[band_idx] += np.sum(band_data)
                                sum_sq_values_landsat[band_idx] += np.sum(band_data**2)
                                count_values_landsat[band_idx] += len(band_data)
                            else:
                                print(f"[DEBUG STATS] Warning: Band {band_idx} in {path_landsat} has no valid data after clipping/nan_to_num for stats.")
                        num_landsat_samples_for_stats += 1
                    except Exception as e:
                        print(f"     [STATS ERROR] Error processing {path_landsat} for Landsat stats: {e}")


        # *** MODIFICA: Il dizionario `stats` ora ha chiavi separate per ERA5 raster e tabulare ***
        stats = {
            'mean_sentinel': torch.zeros(N_BANDS_SENTINEL, dtype=torch.float32),
            'std_sentinel': torch.zeros(N_BANDS_SENTINEL, dtype=torch.float32),
            'mean_landsat': torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float32),
            'std_landsat': torch.zeros(N_BANDS_LANDSAT_ACTUAL, dtype=torch.float32),
            'mean_dem': torch.zeros(1, dtype=torch.float32),
            'std_dem': torch.zeros(1,dtype=torch.float32),
            'mean_era5_raster': torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float32), # NUOVA CHIAVE
            'std_era5_raster': torch.zeros(N_BANDS_ERA5_RASTER, dtype=torch.float32),   # NUOVA CHIAVE
            'mean_era5_tabular': torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float32), # NUOVA CHIAVE
            'std_era5_tabular': torch.zeros(N_BANDS_ERA5_TABULAR, dtype=torch.float32)  # NUOVA CHIAVE
        }

        # Stats computation for Sentinel (NESSUNA MODIFICA)
        for band_idx in range(N_BANDS_SENTINEL):
            if count_values_sentinel[band_idx] > 0:
                mean_val = sum_values_sentinel[band_idx] / count_values_sentinel[band_idx]
                variance_val = (sum_sq_values_sentinel[band_idx] / count_values_sentinel[band_idx]) - (mean_val**2)
                if variance_val < 0: variance_val = torch.tensor(0.0, dtype=torch.float64)
                std_val = torch.sqrt(variance_val)
                if std_val < 1e-6: std_val = torch.tensor(1e-6, dtype=torch.float64)
                stats['mean_sentinel'][band_idx] = mean_val.to(torch.float32)
                stats['std_sentinel'][band_idx] = std_val.to(torch.float32)
            else:
                stats['mean_sentinel'][band_idx] = 0.0
                stats['std_sentinel'][band_idx] = 1.0
                print(f"     [WARNING] No valid data found for Sentinel band {band_idx} across all samples for stats. Using fallback (0 mean, 1 std).")

        # Stats computation for Landsat (NESSUNA MODIFICA)
        for band_idx in range(N_BANDS_LANDSAT_ACTUAL):
            if count_values_landsat[band_idx] > 0:
                mean_val = sum_values_landsat[band_idx] / count_values_landsat[band_idx]
                variance_val = (sum_sq_values_landsat[band_idx] / count_values_landsat[band_idx]) - (mean_val**2)
                if variance_val < 0: variance_val = torch.tensor(0.0, dtype=torch.float64)
                std_val = torch.sqrt(variance_val)
                if std_val < 1e-6: std_val = torch.tensor(1e-6, dtype=torch.float64)
                stats['mean_landsat'][band_idx] = mean_val.to(torch.float32)
                stats['std_landsat'][band_idx] = std_val.to(torch.float32)
            else:
                stats['mean_landsat'][band_idx] = 0.0
                stats['std_landsat'][band_idx] = 1.0
                print(f"     [WARNING] No valid data found for Landsat band {band_idx} across all samples for stats. Using fallback (0 mean, 1 std).")

        # Stats computation for DEM (NESSUNA MODIFICA)
        if num_dem_samples_for_stats > 0:
            if count_values_dem > 0:
                mean_val = sum_values_dem / count_values_dem
                variance_val = (sum_sq_values_dem / count_values_dem) - (mean_val**2)
                if variance_val < 0: variance_val = torch.tensor(0.0, dtype=torch.float64)
                std_val = torch.sqrt(variance_val)
                if std_val < 1e-6: std_val = torch.tensor(1e-6, dtype=torch.float64)
                stats['mean_dem'] = mean_val.to(torch.float32)
                stats['std_dem'] = std_val.to(torch.float32)
            else:
                stats['mean_dem'] = 0.0
                stats['std_dem'] = 1.0
                print(f" [WARNING] No valid data found for dem across all samples for stats. Using fallback (0 mean, 1 std).")

        # *** MODIFICA: Stats computation for ERA5 (separato per raster e tabulare) ***
        for band_idx in range(N_BANDS_ERA5_RASTER):
            if count_values_era5_raster[band_idx] > 0:
                mean_val = sum_values_era5_raster[band_idx] / count_values_era5_raster[band_idx]
                variance_val = (sum_sq_values_era5_raster[band_idx] / count_values_era5_raster[band_idx]) - (mean_val**2)
                if variance_val < 0: variance_val = torch.tensor(0.0, dtype=torch.float64)
                std_val = torch.sqrt(variance_val)
                if std_val < 1e-6: std_val = torch.tensor(1e-6, dtype=torch.float64)
                stats['mean_era5_raster'][band_idx] = mean_val.to(torch.float32)
                stats['std_era5_raster'][band_idx] = std_val.to(torch.float32)
            else:
                stats['mean_era5_raster'][band_idx] = 0.0
                stats['std_era5_raster'][band_idx] = 1.0
                print(f"     [WARNING] No valid data found for ERA5 Raster band (index {band_idx}) for stats. Using fallback (0 mean, 1 std).")
        
        for band_idx in range(N_BANDS_ERA5_TABULAR):
            if num_era5_samples_for_stats > 0: # Qui usiamo il conteggio dei campioni, non dei pixel
                mean_val = sum_values_era5_tabular[band_idx] / num_era5_samples_for_stats
                variance_val = (sum_sq_values_era5_tabular[band_idx] / num_era5_samples_for_stats) - (mean_val**2)
                if variance_val < 0: variance_val = torch.tensor(0.0, dtype=torch.float64)
                std_val = torch.sqrt(variance_val)
                if std_val < 1e-6: std_val = torch.tensor(1e-6, dtype=torch.float64)
                stats['mean_era5_tabular'][band_idx] = mean_val.to(torch.float32)
                stats['std_era5_tabular'][band_idx] = std_val.to(torch.float32)
            else:
                stats['mean_era5_tabular'][band_idx] = 0.0
                stats['std_era5_tabular'][band_idx] = 1.0
                print(f"     [WARNING] No valid data found for ERA5 Tabular band (index {band_idx}) for stats. Using fallback (0 mean, 1 std).")

        print("Global mean and standard deviation computed successfully!")
        print(f"Final Global Means Sentinel ({num_sentinel_samples_for_stats} samples): {stats['mean_sentinel']}")
        print(f"Final Global STDs Sentinel ({num_sentinel_samples_for_stats} samples): {stats['std_sentinel']}")
        print(f"Final Global Means Landsat ({num_landsat_samples_for_stats} samples): {stats['mean_landsat']}")
        print(f"Final Global STDs Landsat ({num_landsat_samples_for_stats} samples): {stats['std_landsat']}")
        print(f"Final Global Mean DEM ({num_dem_samples_for_stats} samples): {stats['mean_dem']}")
        print(f"Final Global STD DEM ({num_dem_samples_for_stats} samples): {stats['std_dem']}")
        # *** MODIFICA: Messaggi di stampa aggiornati ***
        print(f"Final Global Means ERA5 Raster ({num_era5_samples_for_stats} samples): {stats['mean_era5_raster']}")
        print(f"Final Global STDs ERA5 Raster ({num_era5_samples_for_stats} samples): {stats['std_era5_raster']}")
        print(f"Final Global Means ERA5 Tabular ({num_era5_samples_for_stats} samples): {stats['mean_era5_tabular']}")
        print(f"Final Global STDs ERA5 Tabular ({num_era5_samples_for_stats} samples): {stats['std_era5_tabular']}")
        return stats

    def _normalize_bands_global(self, image: torch.Tensor, sensor_type: str) -> torch.Tensor:
        """
        Normalizes a multi-channel image tensor using global mean and standard deviation.
        Expected input image shape: (C, H, W).
        Applies normalization only to the actual data channels, not to the flag channel.
        """
        if self.global_stats is None:
            return image

        if sensor_type == 'sentinel':
            mean_key = 'mean_sentinel'
            std_key = 'std_sentinel'
            num_bands_to_normalize = N_BANDS_SENTINEL
        elif sensor_type == 'landsat':
            mean_key = 'mean_landsat'
            std_key = 'std_landsat'
            num_bands_to_normalize = N_BANDS_LANDSAT_ACTUAL # Normalizza solo le bande dati
        else:
            raise ValueError("sensor_type must be 'sentinel' or 'landsat'")

        # Estrai le statistiche per le bande da normalizzare
        base_mean = self.global_stats[mean_key][:num_bands_to_normalize].to(image.device).view(-1, 1, 1)
        base_std = self.global_stats[std_key][:num_bands_to_normalize].to(image.device).view(-1, 1, 1)

        # Applica normalizzazione solo alle bande dati
        normalized_data_bands = (image[:num_bands_to_normalize, :, :] - base_mean) / (base_std + 1e-6) 

        # Se ci sono bande extra (come la banda di flag per Landsat), concatenale senza normalizzarle
        if image.shape[0] > num_bands_to_normalize:
            # Assumiamo che le bande extra siano le ultime
            remaining_channels = image[num_bands_to_normalize:, :, :]
            normalized_image = torch.cat([normalized_data_bands, remaining_channels], dim=0)
        else:
            normalized_image = normalized_data_bands

        return normalized_image
    def _normalize_era5_global(self, era5: torch.Tensor) -> torch.Tensor:
        if self.global_stats is None:
            return era5
        mean_key = 'mean_era5_raster'
        std_key = 'std_era5_raster'
        num_bands_to_normalize = N_BANDS_ERA5_RASTER
        # Estrai le statistiche per le bande da normalizzare
        base_mean = self.global_stats[mean_key][:num_bands_to_normalize].to(era5.device).view(-1, 1, 1)
        base_std = self.global_stats[std_key][:num_bands_to_normalize].to(era5.device).view(-1, 1, 1)

        # Applica normalizzazione solo alle bande dati
        normalized_data_bands = (era5[:num_bands_to_normalize, :, :] - base_mean) / (base_std + 1e-6) 

        normalized_image = normalized_data_bands

        return normalized_image
    
    def _normalize_era5_tabular(self, era5_tabular_bands_spatial_means: torch.Tensor) -> torch.Tensor:
        """
        Normalizes ERA5 tabular bands (e.g., temperature, pressure) using sample-wise global stats.
        Assumes era5_tabular_bands_spatial_means is of shape (C_tabular,).
        """
        if self.global_stats is None:
            return era5_tabular_bands_spatial_means

        mean_key = 'mean_era5_tabular'
        std_key = 'std_era5_tabular'
        num_bands_to_normalize = N_BANDS_ERA5_TABULAR

        # Estrai le statistiche per le bande tabulari
        # Assicurati che le dimensioni siano corrette: (C_tabular,) per broadcasting con il tensore di input
        base_mean = self.global_stats[mean_key].to(era5_tabular_bands_spatial_means.device)
        base_std = self.global_stats[std_key].to(era5_tabular_bands_spatial_means.device)

        # Applica normalizzazione
        normalized_tabular_bands = (era5_tabular_bands_spatial_means - base_mean) / (base_std + 1e-6)
        return normalized_tabular_bands
    
    def _normalize_dem_global(self, dem: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a DEM tensor using global mean and standard deviation.
        Expected input shape: (1, H, W).
        """
        if self.global_stats is None:
            return dem

        mean = self.global_stats['mean_dem'].to(dem.device).view(1, 1, 1)
        std = self.global_stats['std_dem'].to(dem.device).view(1, 1, 1)

        normalized_dem = (dem - mean) / (std + 1e-6)
        return normalized_dem

    def _read_image(self, path: str) -> np.ndarray:
        """Helper to read a single TIFF image (12 or 15 bands) using rasterio."""
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32) # Read all bands (C, H, W)
            # Transpose to (H, W, C) for Albumentations
            img = np.transpose(img, (1, 2, 0))
            return img

    def _read_mask(self, path: str) -> np.ndarray:
        """Helper to read a single TIFF mask using rasterio."""
        with rasterio.open(path) as src:
            mask = src.read(1)
            mask = np.nan_to_num(mask, nan=0).astype(np.uint8)
            mask = (mask > 0).astype(np.uint8)
            return mask
    def _read_streets(self,path: str) -> np.ndarray:
        """Helper to read a single TIFF streets image using rasterio"""
        with rasterio.open(path) as src:
            streets = src.read(1)
            streets = np.nan_to_num(streets, nan=0).astype(np.uint8)
            return streets
    def _read_dem(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            dem = src.read(1)
            dem = np.nan_to_num(dem, nan=0).astype(np.float32)
            return dem
    def _map_samples_to_bins(self):
        print("Mappatura dei campioni ai bin di dimensione...")
        for idx in range(len(self.fire_dirs)):
            fire_dir = self.fire_dirs[idx]
            files = os.listdir(fire_dir)
            gt_mask_path = None
            for f in files:
                if "GTSentinel" in f and f.endswith(".tif"):
                    gt_mask_path = os.path.join(fire_dir, f)
                    break
            
            if gt_mask_path is None:
                print(f"Attenzione: Nessuna maschera GT trovata in {fire_dir}. Salto questo campione.")
                # Se un campione non ha una maschera valida, puoi gestirlo qui.
                # Per ora lo mappiamo a un bin fittizio o lo saltiamo.
                self.sample_pixel_counts[idx] = 0
                self.sample_bins[idx] = -1
                continue

            gt_mask_np = self._read_mask(gt_mask_path)
            pixel_count = np.sum(gt_mask_np > 0)
            self.sample_pixel_counts[idx] = pixel_count
            
            # Trova l'indice del bin
            bin_idx = 0
            while bin_idx < len(self.bin_thresholds) and pixel_count > self.bin_thresholds[bin_idx]:
                bin_idx += 1
            self.sample_bins[idx] = bin_idx
        
        # Stampa le statistiche per verificare la distribuzione
        print("Distribuzione dei campioni nei bin:")
        bin_counts = {i: 0 for i in range(len(self.bin_thresholds) + 1)}
        for bin_idx in self.sample_bins.values():
            if bin_idx != -1: # Ignora i campioni saltati
                bin_counts[bin_idx] += 1
        print(bin_counts)
    def __len__(self):
        """Returns the total number of valid fire event directories in the dataset."""
        return len(self.fire_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, 
                                             torch.Tensor, 
                                             torch.Tensor, 
                                             torch.Tensor,
                                             torch.Tensor, 
                                             torch.Tensor,
                                             torch.Tensor]: 
        fire_dir = self.fire_dirs[idx]
        files = os.listdir(fire_dir)
        
        sample_type = self.sample_type_map.get(fire_dir, 'sentinel_only') # Default a 'sentinel_only' se non trovato
        
        # --- SENTINEL LOADING (always present) ---
        best_image_info = find_best_image_in_folder(fire_dir)
        if best_image_info is None:
             # Questo non dovrebbe succedere dopo il filtering, ma per robustezza
            raise FileNotFoundError(f"Nessuna immagine Sentinel valida con CM trovata per '{os.path.basename(fire_dir)}'. Impossibile caricare il campione.")

        pre_sentinel_image_path = best_image_info['sentinel_path']
        
        # Leggi l'immagine Sentinel originale e puliscila
        # Non è più necessario usare read_image_and_metadata qui, _read_image basta
        pre_sentinel_np_raw = self._read_image(pre_sentinel_image_path)
        pre_sentinel_np = np.nan_to_num(pre_sentinel_np_raw, nan=0.0, posinf=0.0, neginf=0.0)
        pre_sentinel_np = np.clip(pre_sentinel_np, 0.0, 10000.0)

        # --- LANDSAT LOADING (condizionale e con banda di flag) ---
        landsat_data_present = False # Flag per indicare se Landsat è stato effettivamente caricato
        landsat_resampled_np = np.zeros((self.target_size[0], self.target_size[1], N_BANDS_LANDSAT_ACTUAL), dtype=np.float32)
        # La banda di flag inizialmente a zero
        landsat_presence_flag = np.zeros((self.target_size[0], self.target_size[1], 1), dtype=np.float32)

        ## STREETS LOADING
        streets_path = None
        for f in files:
            if f.endswith("streets.tif"):
                streets_path = os.path.join(fire_dir,f)
                break
        if streets_path is None:
            streets_np = np.zeros((self.target_size[0], self.target_size[1], 1), dtype=np.float32)
        else:
            streets_np = self._read_streets(streets_path)
            #streets_np = streets_np[..., np.newaxis].astype(np.float32)
            streets_np = streets_np.astype(np.float32)
        ## DEM LOADING
        dem_path = None
        for f in files:
            if "dem_10m" in f and f.endswith(".tif"):
                dem_path = os.path.join(fire_dir,f)
                break
        if dem_path is None:
            dem_np = np.zeros((self.target_size[0], self.target_size[1], 1), dtype= np.float32)
        else:
            dem_np_raw = self._read_dem(dem_path)
            dem_np = np.nan_to_num(dem_np_raw, nan=0.0, posinf=0.0, neginf=0.0)
            #dem_np = dem_np[..., np.newaxis].astype(np.float32)
        
        ## ERA5 LOADING
        era5_path = None
        for f in files:
            if "era5" in f and f.endswith(".tif"):
                era5_path = os.path.join(fire_dir,f)
                break

        # Inizializza numpy arrays per le bande ERA5 raster e tabulari
        # Le dimensioni per raster saranno (H, W, C_raster)
        era5_raster_np = np.zeros((self.target_size[0], self.target_size[1], N_BANDS_ERA5_RASTER), dtype=np.float32)
        # Le "bande" tabulari inizialmente sono ancora (H,W,C_tabular) prima della media spaziale
        era5_tabular_np = np.zeros((self.target_size[0], self.target_size[1], N_BANDS_ERA5_TABULAR), dtype=np.float32)

        if era5_path is None:
            print(f"Warning: ERA5 data not found for {fire_dir}. Using zero arrays for ERA5.")
            # I default (array di zeri) sono già impostati sopra
        else:
            try:
                era5_np_raw = self._read_image(era5_path)
                era5_np_full = np.nan_to_num(era5_np_raw, nan=0.0, posinf=0.0, neginf=0.0)
                era5_np_full = np.clip(era5_np_full, 0.0, 10000.0)

                if era5_np_full.shape[2] != N_BANDS_ERA5:
                    print(f"Warning: ERA5 file {era5_path} has {era5_np_full.shape[2]} bands, expected {N_BANDS_ERA5}. Using zero arrays for ERA5.")
                else:
                    # Estrai le bande raster e tabulari dal numpy array completo
                    era5_raster_np = era5_np_full[:, :, ERA5_RASTER_BAND_INDICES]
                    era5_tabular_np = era5_np_full[:, :, ERA5_TABULAR_BAND_INDICES]
            except Exception as e:
                print(f"Error reading ERA5 data from {era5_path}: {e}. Using zero arrays.")
                # Array di zeri rimarranno in caso di errore
        era5_tabular_tensor_raw = torch.from_numpy(era5_tabular_np).permute(2, 0, 1).float() # Converti a (C_tabular, H, W)

        
        ## IGNITION POINT LOADING
        
        ignition_path=None
        for f in files:
            if f.endswith("ignition_pt.tif"):
                ignition_path= os.path.join(fire_dir,f)
                break
        if ignition_path is None:
            ignition_np= np.zeros((self.target_size[0], self.target_size[1], 1), dtype = np.float32) # Ignition not present
        else:
            ignition_np = self._read_mask(ignition_path)
        
        ## LANDCOVER
        landcover_path=None
        for f in files:
            if f.endswith("landcover.tif"):
                landcover_path= os.path.join(fire_dir,f)
                break
        if landcover_path is None:
            raise FileNotFoundError(f"No Landcover mask found in {fire_dir} for index {idx}. This should not happen after filtering.")
        else:
            landcover_np = self._read_streets(landcover_path)
        
        ## LANDSAT LOADING (if present)    
        
        if sample_type == 'both':
            landsat_10m_image_path = None
            for f in files:
                # QUI: Cerca il file Landsat già risampled a 10m
                if "pre_landsat" in f and "_10m.tif" in f and "_CM" not in f:
                    landsat_10m_image_path = os.path.join(fire_dir, f)
                    break
            
            if landsat_10m_image_path and os.path.exists(landsat_10m_image_path):
                try:
                    # Semplicemente leggi l'immagine Landsat a 10m pre-processata
                    landsat_resampled_np_raw = self._read_image(landsat_10m_image_path)
                    landsat_resampled_np = np.nan_to_num(landsat_resampled_np_raw, nan=0.0, posinf=0.0, neginf=0.0)
                    landsat_data_present = True
                    landsat_presence_flag = np.ones((self.target_size[0], self.target_size[1], 1), dtype=np.float32) # Landsat presente
                except Exception as e:
                    print(f"Warning: Error reading pre-resampled Landsat image for {fire_dir}: {e}. Treating as 'sentinel_only'.")
                    # Se la lettura fallisce, i default (array di zeri e flag a zero) rimangono validi
            else:
                # Questo caso può succedere se sample_type_map dice 'both' ma il file 10m non c'è (es. errore pre-processing)
                print(f"Warning: sample_type was 'both' for {fire_dir}, but pre-resampled 10m Landsat image not found at {landsat_10m_image_path}. Treating as 'sentinel_only'.")
        # else: (sample_type == 'sentinel_only') i default (array di zeri e flag a zero) rimangono validi

        # Concatenare la banda di flag ai dati Landsat
        landsat_input_to_aug = np.concatenate([landsat_resampled_np, landsat_presence_flag], axis=2)
        
        # --- GROUND TRUTH LOADING ---
        gt_mask_path = None
        for f in files:
            if "GTSentinel" in f and f.endswith(".tif"):
                gt_mask_path = os.path.join(fire_dir, f)
                break
        if gt_mask_path is None: 
            raise FileNotFoundError(f"No GTSentinel mask found in {fire_dir} for index {idx}. This should not happen after filtering.")
            
        gt_mask_np = self._read_mask(gt_mask_path)
        
        # --- APPLICAZIONE AUGMENTATIONS E NORMALIZZAZIONE ---
        if self.apply_augmentations:
            
            # Applica le trasformazioni a entrambe le immagini e alla maschera
            transformed = self.augmentor(image=pre_sentinel_np, 
                                        landsat_image=landsat_input_to_aug,
                                        streets_image=streets_np,
                                        dem_image=dem_np,
                                        ignition_pt=ignition_np,
                                        era5_image=era5_raster_np,
                                        landcover_np=landcover_np,
                                        mask=gt_mask_np) 
            sentinel_image_tensor = transformed['sentinel_image']
            landsat_image_tensor = transformed['landsat_image'] # Questo ora ha la banda extra
            streets_tensor = transformed['streets_image']
            dem_tensor = transformed['dem_image']
            era5_tensor = transformed['era5_image']
            ignition_tensor = transformed['ignition_pt']
            gt_mask_tensor = transformed['mask']
            landcover_tensor = transformed['landcover_np']
            

        else:
             ## validation (only to Tensor, no aug)
            transformed_sentinel = self.eval_transform(image=pre_sentinel_np, mask=gt_mask_np)
            sentinel_image_tensor = transformed_sentinel['image']
            gt_mask_tensor = transformed_sentinel['mask']

            # Per la trasformazione di valutazione, applica ToTensorV2 all'array NumPy già concatenato
            transformed_landsat = self.eval_transform(image=landsat_input_to_aug)
            landsat_image_tensor = transformed_landsat['image'] # Questo ora ha la banda extra

            transformed_streets = self.eval_transform(image=streets_np)
            streets_tensor = transformed_streets['image']

            transformed_dem = self.eval_transform(image=dem_np)
            dem_tensor = transformed_dem['image']

            transformed_era5 = self.eval_transform(image=era5_raster_np)
            era5_tensor = transformed_era5['image']
            
            transformed_ignition_pt= self.eval_transform(image=ignition_np)
            ignition_tensor= transformed_ignition_pt['image']

            transformed_landcover = self.eval_transform(image=landcover_np)
            landcover_tensor = transformed_landcover['image']
    
        # Global normalization (se global_stats è disponibile)
        if self.global_stats:
            #sentinel_image_tensor = sentinel_image_tensor.float() / 10000.0
            sentinel_image_tensor = self._normalize_bands_global(sentinel_image_tensor, sensor_type='sentinel')
            #landsat_image_tensor = landsat_image_tensor.float() / 65535.0
            landsat_image_tensor = self._normalize_bands_global(landsat_image_tensor, sensor_type='landsat')
            dem_tensor = self._normalize_dem_global(dem_tensor)
            era5_tensor = self._normalize_era5_global(era5_tensor)
            era5_tabular_spatial_means = torch.mean(era5_tabular_tensor_raw, dim=(-1, -2))
            era5_tabular = self._normalize_era5_tabular(era5_tabular_spatial_means).float()
            streets_tensor = streets_tensor.float() / 5.0 
            other_data_tensor = torch.concat([streets_tensor, dem_tensor], dim=0)
            ignition_tensor = ignition_tensor.float()
        
        other_data_tensor = torch.concat([dem_tensor,streets_tensor], dim=0) # DEM,Streets, ignition point in a single tensor

        return sentinel_image_tensor, landsat_image_tensor, other_data_tensor,ignition_tensor,era5_tensor,era5_tabular, landcover_tensor, gt_mask_tensor

    def get_sample_info(self, idx: int) -> dict:
        """
        Returns a dictionary containing information about the sample at the given index.
        """
        fire_dir = self.fire_dirs[idx]
        dir_name = os.path.basename(fire_dir)
        match = re.search(r'fire_(\d+)', dir_name)
        fire_id = int(match.group(1)) if match else None

        # Utilizza find_best_image_in_folder per ottenere il nome del file corretto
        best_image_info = find_best_image_in_folder(fire_dir)
        pre_image_file_sentinel = os.path.basename(best_image_info['sentinel_path']) if best_image_info else "N/A"

        pre_image_file_landsat = "N/A"
        files = os.listdir(fire_dir)
        for f in files:
            # QUI: Cerca il Landsat 10m per il sample info
            if "pre_landsat" in f and "_10m.tif" in f and "_CM" not in f:
                pre_image_file_landsat = f
                break

        gt_mask_file = "N/A"
        for f in files:
            if "GTSentinel" in f and f.endswith(".tif"):
                gt_mask_file = f
                break

        return {
            'fire_id': fire_id,
            'directory': fire_dir,
            'pre_image_file_sentinel': pre_image_file_sentinel,
            'pre_image_file_landsat': pre_image_file_landsat, 
            'gt_mask_file': gt_mask_file,
            'fire_date': self.fire_id_to_date_map.get(fire_id, "N/A"),
            'sample_type': self.sample_type_map.get(fire_dir, 'unknown') 
        }