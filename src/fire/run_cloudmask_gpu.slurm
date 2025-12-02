#!/bin/bash
#SBATCH --job-name=era5_parallel     # Nome del job
#SBATCH --output=logs/era5_parallel_%A_%a.out   # Output del job array (%A=JobID, %a=ArrayTaskID)
#SBATCH --error=logs/era5_parallel_%A_%a.err    # Errore del job array
#SBATCH --ntasks=1                   # Ogni task processerà un chunk
#SBATCH --cpus-per-task=4            # Mantiene 4 CPU per l'elaborazione interna di un folder
#SBATCH --gpus=0                     # ERA5 non usa GPU, rimuovi la richiesta per accedere a più nodi CPU
#SBATCH --mem=16G                    # Memoria RAM
#SBATCH --time=34:00:00              # Tempo massimo per CIASCUN task dell'array (es. 34 ore)
#SBATCH --array=0-1  

# Rendi disponibili i comandi 'module'
source /etc/profile

# Attiva l'ambiente virtuale (da cambiare col percorso esatto)
source ~/thesis-wildfire-danger-bavaro/.venv/bin/activate

# Esegui lo script
python src/fire/generateEra5.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_COUNT}"
