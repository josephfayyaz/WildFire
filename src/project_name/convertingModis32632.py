import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio

input_path = "piedmont/fire_5440/fire_5440_2017-10-24_modis_Emis.tif"
output_path = "piedmont/fire_5440/fire_5440_2017-10-24_modis_Emis_32632_matched.tif"
target_crs = "EPSG:32632"

with rasterio.open(input_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'dtype': src.dtypes[0],       # Mantieni il tipo originale
        'nodata': 0                   # Mantieni nodata se coerente
    })

    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

'''
def convert_modis_crs(input_path, output_path, target_crs='EPSG:32632'):
    """
    Converte un raster MODIS dal CRS sinusoidale a un CRS di destinazione (default: EPSG:32632).

    Args:
        input_path (str): Percorso del file TIFF MODIS in CRS sinusoidale.
        output_path (str): Percorso dove salvare il file TIFF riproiettato.
        target_crs (str): Codice EPSG del sistema di coordinate di destinazione.
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'nodata':0,
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32'
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # Puoi provare anche altri metodi di resampling
                )
    print(f"File riproiettato e salvato in: {output_path}")

# Esempio di utilizzo (assicurati di avere il percorso corretto del tuo file TIFF MODIS)
input_tif_path = "piedmont/fire_5440/fire_5440_2017-10-24_modis_Emis.tif"
output_tif_path = "piedmont/fire_5440/fire_5440_2017-10-24_modis_Emis_32632_corrected.tif"
convert_modis_crs(input_tif_path, output_tif_path)
'''