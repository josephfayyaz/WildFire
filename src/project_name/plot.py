import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
import matplotlib.patches as mpatches
import os
from PIL import Image

SCALE_FACTOR = 1  # Riduzione risoluzione
DPI = 300         # Risoluzione PNG

# Colori maschera CloudSEN12
INTERPRETATION_CLOUDSEN12 = ["clear", "Thick cloud", "Thin cloud", "Cloud shadow"]
COLORS_CLOUDSEN12 = np.array([
    [139, 64, 0],       # clear 0
    [220, 220, 220],    # Thick cloud 1 
    [180, 180, 180],    # Thin cloud 2
    [60, 60, 60]        # cloud shadow 3
], dtype=np.float32) / 255
class_values_ordered_lulc = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11]

values_to_colors_lulc = {
    0: 'gray',        # No Data
    1: 'lightblue',   # Water
    2: 'forestgreen', # Trees
    4: 'olivedrab',   # Flooded vegetation
    5: 'darkorange',  # Crops
    7: 'peru',        # Built area
    8: 'sienna',      # Bare ground
    9: 'aliceblue',   # Snow/ice
    10: 'lightgrey',  # Clouds
    11: 'khaki'      # Rangeland
}

# Assicurati che ogni valore abbia un nome di classe associato
values_to_classes = {
    0: 'No Data',
    1: 'Water',
    2: 'Trees',
    4: 'Flooded vegetation',
    5: 'Crops',
    7: 'Built area',
    8: 'Bare ground',
    9: 'Snow/ice',
    10: 'Clouds',
    11: 'Rangeland'
}

# Creazione della colormap e della normalizzazione
cmap_lulc = ListedColormap([values_to_colors_lulc[val] for val in class_values_ordered_lulc])
bounds = np.array(class_values_ordered_lulc) - 0.5
norm_lulc = BoundaryNorm(bounds, cmap_lulc.N)


def normalize_band(band):
    vmin = np.nanpercentile(band, 2)
    vmax = np.nanpercentile(band, 98)
    if vmax - vmin == 0:
        return np.zeros_like(band, dtype=np.uint8)
    band_norm = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return (band_norm * 255).astype(np.uint8)

def plot_cloud_mask(tif_path):
    with rasterio.open(tif_path) as src:
        mask = src.read(1).astype(np.uint8)  # Assume una sola banda

    if np.any(mask > 3):
        print(f"[ERRORE] Valori fuori range nella maschera: trovati {np.unique(mask)}")
        return

    colored_mask = (COLORS_CLOUDSEN12[mask] * 255).astype(np.uint8)
    img = Image.fromarray(colored_mask, 'RGB')

    base, _ = os.path.splitext(tif_path)
    out_png_path = base + "_mask.png"
    img.save(out_png_path)
    print(f"[SALVATO] {out_png_path}")

def plot_ignition(tif_path):
    """
    Carica un TIFF della mappa di distanza dell'ignition point (float tra 0 e 1)
    e lo visualizza usando una colormap. Salva anche un'immagine PNG.
    """
    try:
        with rasterio.open(tif_path) as src:
            # Leggi la banda, assicurati che sia float (dovrebbe già esserlo)
            distance_map = src.read(1).astype(np.float32)
            
            # Stampa alcune statistiche per verifica
            print(f"  Statistiche mappa di distanza per {os.path.basename(tif_path)}:")
            print(f"    Min: {np.min(distance_map):.4f}")
            print(f"    Max: {np.max(distance_map):.4f}")
            print(f"    Media: {np.mean(distance_map):.4f}")
            
            # Creazione della figura e dell'asse per la visualizzazione
            fig, ax = plt.subplots(figsize=(10, 10)) # Dimensione per il plot

            # Visualizza la mappa di distanza
            # Usiamo 'hot' o 'inferno' per una mappa di calore dove il punto caldo è il centro.
            # 'vmin' e 'vmax' assicurano che la colormap sia mappata sull'intero range [0, 1]
            # 'interpolation='none'' (o 'nearest') per pixel discreti, 'bilinear' per più sfumato.
            # Visto che è una mappa di distanza, un'interpolazione più sfumata ha senso.
            im = ax.imshow(distance_map, cmap='inferno', vmin=0, vmax=1, interpolation='none')
            
            # Aggiungi una colorbar per interpretare i valori
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Intensità (Distanza dal punto di Ignizione)')

            ax.set_title(f"Mappa di Distanza Ignition Point: {os.path.basename(tif_path)}", fontsize=14)
            ax.axis('off') # Nasconde gli assi per una visualizzazione più pulita

            # Salva il plot come PNG
            base, _ = os.path.splitext(tif_path)
            out_png_path = base + "_distance_map.png" # Nome file più specifico
            plt.savefig(out_png_path, bbox_inches='tight', dpi=DPI, pad_inches=0)
            
            print(f"[SALVATO] {out_png_path}")
            
            plt.close(fig) # Chiudi la figura per liberare memoria
            return True

    except Exception as e:
        print(f"❌ Errore durante la visualizzazione di '{tif_path}': {e}")
        return False

def plot_sentinel(tif_path):
    
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        new_height = height // SCALE_FACTOR
        new_width = width // SCALE_FACTOR

        band_descriptions = src.descriptions
        try:
            b04_index = band_descriptions.index("B04") + 1
            b08_index = band_descriptions.index("B08") + 1
            b12_index = band_descriptions.index("B12") + 1
        except ValueError:
            print(f"[ERRORE] Bande B04, B08 o B12 mancanti in {tif_path}")
            return

        b04 = src.read(b04_index, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
        b08 = src.read(b08_index, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
        b12 = src.read(b12_index, out_shape=(new_height, new_width), resampling=Resampling.bilinear)

    # Normalizza
    red   = normalize_band(b12)
    green = normalize_band(b08)
    blue  = normalize_band(b04)
    rgb_image = np.dstack((red, green, blue))

    # Output path
    base, ext = os.path.splitext(tif_path)
    output_path = base + "_png.png"

    # Salva PNG
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("Composizione falsi colori (SWIR/NIR/RED)", fontsize=14)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SALVATO] {output_path}")

def plot_landsat(tif_path):
    
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        new_height = height // SCALE_FACTOR
        new_width = width // SCALE_FACTOR

        band_descriptions = src.descriptions
        try:
            red_idx   = band_descriptions.index("red") + 1
            nir_idx   = band_descriptions.index("nir08") + 1
            swir_idx  = band_descriptions.index("swir16") + 1
        except ValueError:
            print(f"[ERRORE] Bande red, nir08 o swir6 mancanti in {tif_path}")
            print("Bande trovate:", band_descriptions)
            return

        red  = src.read(red_idx, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
        nir  = src.read(nir_idx, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
        swir = src.read(swir_idx, out_shape=(new_height, new_width), resampling=Resampling.bilinear)

    # Normalizza
    r = normalize_band(swir)
    g = normalize_band(nir)
    b = normalize_band(red)
    rgb_image = np.dstack((r, g, b))

    # Output path
    base, ext = os.path.splitext(tif_path)
    output_path = base + "_png.png"

    # Salva PNG
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("Composizione falsi colori (SWIR/NIR/RED)", fontsize=14)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SALVATO] {output_path}")
def plot_modis(tif_path):
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        new_height = height // SCALE_FACTOR
        new_width = width // SCALE_FACTOR

        band_descriptions = src.descriptions
        try:
            emis31_idx = band_descriptions.index("Emis_31") + 1
            emis32_idx = band_descriptions.index("Emis_32") + 1
        except ValueError:
            print(f"[ERRORE] Bande Emis_31 o Emis_32 mancanti in {tif_path}")
            print("Bande trovate:", band_descriptions)
            return

        emis31 = src.read(emis31_idx, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
        emis32 = src.read(emis32_idx, out_shape=(new_height, new_width), resampling=Resampling.bilinear)

    # Salva immagine Emis_31
    emis31_norm = normalize_band(emis31)
    base, ext = os.path.splitext(tif_path)
    output_31 = base + "_emis31.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(emis31_norm, cmap="inferno")
    plt.axis("off")
    plt.title("MODIS Emissivity 31 (11 µm)", fontsize=14)
    plt.savefig(output_31, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SALVATO] {output_31}")

    # Salva immagine Emis_32
    emis32_norm = normalize_band(emis32)
    output_32 = base + "_emis32.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(emis32_norm, cmap="inferno")
    plt.axis("off")
    plt.title("MODIS Emissivity 32 (12 µm)", fontsize=14)
    plt.savefig(output_32, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SALVATO] {output_32}")
def plot_streets(tif_path):
    """
    Genera una PNG pulita (senza assi o etichette) per il raster delle strade.
    Assegna colori ai valori GP_RTP (1-5) e bianco per il background (0).
    """
    with rasterio.open(tif_path) as src:
        streets_data = src.read(1) # Legge la prima (e unica) banda

    # Definizione della colormap personalizzata
    # Bianco per il valore 0 (No Road)
    # Una scala di colori (es. viridis) per i valori GP_RTP da 1 a 5
    colors = ['#FFFFFF'] + plt.cm.viridis(np.linspace(0, 1, 5)).tolist()
    cmap = ListedColormap(colors)
    
    # Normalizzazione per mappare i valori [0, 1, 2, 3, 4, 5] ai colori
    # vmax = 5 perché i valori GP_RTP vanno fino a 5.
    norm = Normalize(vmin=0, vmax=5) 

    base, ext = os.path.splitext(tif_path)
    output_png_path = base + "_viz.png" # "_viz.png" per distinguere dal tif

    plt.figure(figsize=(streets_data.shape[1]/DPI, streets_data.shape[0]/DPI), dpi=DPI)
    plt.imshow(streets_data, cmap=cmap, norm=norm, interpolation='nearest') # 'nearest' per pixel netti
    
    plt.axis('off') # Nasconde gli assi
    # Rimuovi qualsiasi padding o margine extra
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0) # Salva il PNG senza bordi bianchi
    plt.close()
    print(f"[SALVATO] {output_png_path}")
def plot_dem(tif_path):
    """
    Genera un'immagine PNG del DEM, visualizzando l'elevazione con una colormap.
    """
    with rasterio.open(tif_path) as src:
        dem_data = src.read(1)
        nodata_value = src.nodata if src.nodata is not None else -9999.0
    
    # Sostituisci i nodata con NaN per un calcolo robusto dei percentili
    dem_data_masked = np.copy(dem_data).astype(np.float32)
    dem_data_masked[dem_data_masked == nodata_value] = np.nan

    # Calcola vmin e vmax per lo stretching percentile
    vmin = np.nanpercentile(dem_data_masked[~np.isnan(dem_data_masked)], 2)
    vmax = np.nanpercentile(dem_data_masked[~np.isnan(dem_data_masked)], 98)

    # Gestione caso in cui i dati sono piatti per evitare divisioni per zero
    if vmax - vmin == 0:
        print(f"AVVISO: Dati DEM piatti o uniformi in {tif_path}. Plot potrebbe essere vuoto o non significativo.")
        vmax = vmin + 1.0 # Assicurati che vmax sia maggiore di vmin per la normalizzazione

    base, _ = os.path.splitext(tif_path)
    output_png_path = base + "_plot.png"

    plt.figure(figsize=(10, 10))
    plt.imshow(dem_data_masked, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Elevazione (metri)')
    plt.title("Modello di Elevazione Digitale (DEM)", fontsize=14)
    plt.axis("off")
    plt.savefig(output_png_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SALVATO] {output_png_path}")

def plot_landcover(tif_path):
    """
    Genera un'immagine PNG del Land Cover, visualizzando le classi con una colormap discreta
    e una legenda leggibile.
    """
    with rasterio.open(tif_path) as src:
        lc_data = src.read(1)
        nodata_value = src.nodata if src.nodata is not None else 0 
    
    # Sostituisci i nodata con NaN per evitare che vengano plottati come classe
    lc_data_masked = np.copy(lc_data).astype(np.float32)
    lc_data_masked[lc_data_masked == nodata_value] = np.nan

    base, _ = os.path.splitext(tif_path)
    output_png_path = base + "_plot.png"

    plt.figure(figsize=(12, 12)) 
    
    plt.imshow(lc_data_masked, cmap=cmap_lulc, norm=norm_lulc, interpolation='nearest') 

    plt.title("Land Use / Land Cover (LULC) a 10m", fontsize=14)
    plt.axis("off")

    # Creazione della legenda personalizzata
    patches = [mpatches.Patch(color=values_to_colors_lulc[val], label=values_to_classes[val])
               for val in class_values_ordered_lulc if val in values_to_classes]
    
    # Ordina le patch per un display consistente
    patches.sort(key=lambda x: class_values_ordered_lulc.index([k for k,v in values_to_classes.items() if v == x.get_label()][0]))

    # Posiziona la legenda esternamente al plot
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)

    plt.savefig(output_png_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SALVATO] {output_png_path}")

# ESEMPIO USO
if __name__ == "__main__":
    tif_path = "piedmont_new/fire_6253/fire_6253_landcover.tif"
    if "landsat" in tif_path.lower():
        plot_landsat(tif_path)
    elif "cm" in tif_path.lower():
        plot_cloud_mask(tif_path)
    elif "sentinel" in tif_path.lower():
        plot_sentinel(tif_path)
    elif "modis" in tif_path.lower():
        plot_modis(tif_path)
    elif "_streets.tif" in tif_path.lower(): # Nuova condizione per il file delle strade
        plot_streets(tif_path)
    elif "dem" in tif_path.lower():
        plot_dem(tif_path)
    elif "ignition" in tif_path.lower():
        plot_ignition(tif_path)
    elif "landcover" in tif_path.lower(): # Nuova condizione per il Land Cover
        plot_landcover(tif_path)
