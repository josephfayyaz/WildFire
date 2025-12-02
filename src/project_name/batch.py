import geopandas as gpd
import yaml
import subprocess
import os

CONFIG_PATH = "src/project_name/config.yaml"
SCRIPT_PATH = "src/project_name/satellite_pre.py"
LOG_PATH = "src/project_name/processed_ids_pre.txt"  # file per loggare gli ID processati

# Carica config di base
with open(CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

geojson_path = base_config["geojson_path"]
satellite = base_config["satellite"]
interval = base_config.get("interval", 5)

# GeoDataFrame da cui ottenere tutti gli ID incendio
gdf = gpd.read_file(geojson_path)
fire_ids = gdf["id"].unique()

# Carica gli ID già processati, se esistono
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        processed_ids = set(int(line.strip()) for line in f if line.strip().isdigit())
else:
    processed_ids = set()

# Ordina e processa i fire_id
for fire_id in sorted(fire_ids):
    if fire_id in processed_ids:
        continue

    print(f"🔥 Processing fire_id: {fire_id}")

    config = {
        "fire_id": int(fire_id),
        "satellite": satellite,
        "geojson_path": geojson_path,
        "interval": interval
    }

    # Scrivi nuova config.yaml
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    try:
        # Esegui satellite.py (non stoppa in caso di errore)
        result = subprocess.run(["python", SCRIPT_PATH], capture_output=True, text=True)

        # Output std e errori (per log o debugging)
        print(result.stdout)
        if result.returncode != 0:
            print(f"⚠️  Warning: errore parziale per fire_id {fire_id}")
            print(result.stderr)

    except KeyboardInterrupt:
        print("\n⏹️  Interruzione manuale. Rifaremo il fire_id corrente nel prossimo run.")
        break  # NON loggare questo ID

    # In ogni caso, aggiungi al log se non interrotto da tastiera
    with open(LOG_PATH, "a") as f:
        f.write(f"{fire_id}\n")

