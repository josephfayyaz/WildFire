#!/bin/bash
#SBATCH --job-name=trainingModel                # Nome del job
#SBATCH --output=logs/trainingModel_%j.out      # File di output (%j = job ID)
#SBATCH --error=logs/trainingModel_%j.err       # File di errore
#SBATCH --ntasks=1                          # Numero totale di task (processi)
#SBATCH --cpus-per-task=4                   # CPU per ogni task (thread)
#SBATCH --gpus=1                            # GPU richiesta
#SBATCH --mem=16G                           # Memoria RAM
#SBATCH --time=04:00:00                     # Tempo massimo (hh:mm:ss)
#SBATCH --partition=RTX                      # Partizione del cluster (GPU node)

# Rendi disponibili i comandi 'module'
source /etc/profile


# Attiva l'ambiente virtuale (da cambiare col percorso esatto)
source .venv/bin/activate

# Esegui lo script
python src/project_name/modelWithLandsat/main.py