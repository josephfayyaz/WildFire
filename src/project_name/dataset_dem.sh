#!/bin/bash
#
# SLURM JOB SCRIPT FOR PIEDMONT FIRE DATA PREP

#SBATCH --job-name=PiedmontDataPrep       # Nome del job per la preparazione dati
#SBATCH --output=logs/data_prep_%j.out    # File di output (%j = job ID)
#SBATCH --error=logs/data_prep_%j.err     # File di errore
#SBATCH --ntasks=1                        # Numero totale di task (il tuo script Python è single-process)
#SBATCH --cpus-per-task=4                 # CPU per ogni task (generoso, per eventuali operazioni di I/O o NumPy)
#SBATCH --gpus=0                          # NESSUNA GPU richiesta per la preparazione dati
#SBATCH --mem=16G                         # Memoria RAM (adatta se necessario, 16G è un buon inizio)
#SBATCH --time=4-00:00:00                 # Tempo massimo (gg-hh:mm:ss) - 4 giorni come esempio, adatta


# Rendi disponibili i comandi 'module' (spesso utile su HPC)
source /etc/profile

# Attiva l'ambiente virtuale
# VERIFICA CHE QUESTO PERCORSO SIA ASSOLUTO E CORRETTO per il tuo ambiente dove hai installato le librerie GEO.
source ~/thesis-wildfire-danger-bavaro/.venv/bin/activate

# Esegui lo script Python per la preparazione del dataset
# Assicurati che questo percorso sia corretto rispetto alla directory da cui sottometti il job.
# Se sottometti da '~/thesis-wildfire-danger-bavaro/', allora il percorso è relativo.
python src/project_name/generateDem.py