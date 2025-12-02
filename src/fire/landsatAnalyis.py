import rasterio
import numpy as np
import os

def analyze_landsat_bands(landsat_file_path: str):
    """
    Analizza un file TIFF Landsat e stampa le statistiche per ogni banda.

    Args:
        landsat_file_path (str): Il percorso completo al file .tif di Landsat.
    """
    if not os.path.exists(landsat_file_path):
        print(f"Errore: File non trovato a {landsat_file_path}")
        return

    print(f"\n--- Analisi del file: {os.path.basename(landsat_file_path)} ---")

    try:
        with rasterio.open(landsat_file_path) as src:
            landsat_img = src.read() # Legge tutte le bande (C, H, W)
            profile = src.profile # Ottieni i metadati
            
            print(f"  Shape (C, H, W): {landsat_img.shape}")
            print(f"  Dtype: {landsat_img.dtype}")
            print(f"  Numero di bande: {src.count}")
            print(f"  Larghezza (pixels): {src.width}")
            print(f"  Altezza (pixels): {src.height}")
            print(f"  CRS: {src.crs}")
            print(f"  Trasformazione: {src.transform}")

            print("\n  Statistiche per tutte le bande:")
            total_nans = np.sum(np.isnan(landsat_img))
            total_infs = np.sum(np.isinf(landsat_img))
            print(f"    Totale NaNs: {total_nans}")
            print(f"    Totale Infs: {total_infs}")
            
            # Calcola le statistiche generali solo sui pixel validi per un overview
            valid_pixels_overall = landsat_img[~np.isnan(landsat_img) & ~np.isinf(landsat_img)]
            if len(valid_pixels_overall) > 0:
                print(f"    Minimo globale (pixel validi): {np.min(valid_pixels_overall):.2f}")
                print(f"    Massimo globale (pixel validi): {np.max(valid_pixels_overall):.2f}")
                print(f"    Media globale (pixel validi): {np.mean(valid_pixels_overall):.2f}")
                print(f"    Mediana globale (pixel validi): {np.median(valid_pixels_overall):.2f}")
            else:
                print("    Nessun pixel valido trovato nell'immagine complessiva.")


            print("\n  Statistiche dettagliate per ciascuna banda:")
            # Se conosci i nomi delle bande, puoi associarli qui.
            # Esempio: band_names = ['atran', 'blue', ..., 'urad']
            # Se non li conosci, userà solo il numero di banda.
            band_names = ['atran', 'blue', 'cdist', 'coastal', 'drad', 'emis', 'emsd', 'green', 'lwir11', 'nir08', 'red', 'swir16', 'swir22', 'trad', 'urad']
            
            for i in range(src.count):
                band_data = landsat_img[i, :, :] # Seleziona la banda i
                band_name = band_names[i] if i < len(band_names) else f"Banda {i+1}"

                valid_band_pixels = band_data[~np.isnan(band_data) & ~np.isinf(band_data)]

                print(f"    {band_name}:")
                print(f"      Numero di NaNs: {np.sum(np.isnan(band_data))}")
                print(f"      Numero di Infs: {np.sum(np.isinf(band_data))}")
                
                if len(valid_band_pixels) > 0:
                    print(f"      Min: {np.min(valid_band_pixels):.4f}")
                    print(f"      Max: {np.max(valid_band_pixels):.4f}")
                    print(f"      Mean: {np.mean(valid_band_pixels):.4f}")
                    print(f"      Median: {np.median(valid_band_pixels):.4f}")
                else:
                    print("      Nessun pixel valido trovato in questa banda.")

    except rasterio.errors.RasterioIOError as e:
        print(f"Errore di lettura del file con Rasterio: {e}")
    except Exception as e:
        print(f"Si è verificato un errore inatteso: {e}")

# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Sostituisci questo percorso con il percorso effettivo a uno dei tuoi file Landsat
    # Esempio per il file che hai mostrato:
    # BASE_DIR = "C:/Users/tuo_utente/piedmont_new" # O la tua base directory
    # fire_id = "5411"
    # landsat_filename = "fire_5411_2017-09-20_pre_landsat_3.tif"
    # landsat_file = os.path.join(BASE_DIR, f"fire_{fire_id}", landsat_filename)

    # Esempio con un percorso fittizio, assicurati di cambiarlo!
    landsat_file_example = "piedmont_new/fire_5497/fire_5497_2017-11-16_pre_landsat_1.tif" 
    
    # Scegli altri file per avere un'idea più completa, es:
    # landsat_file_example_2 = "path/to/another/fire_ID/pre_landsat_X.tif"
    # landsat_file_example_3 = "path/to/yet/another/fire_ID/pre_landsat_Y.tif"

    analyze_landsat_bands(landsat_file_example)
    # analyze_landsat_bands(landsat_file_example_2) # Scommenta per analizzare più file