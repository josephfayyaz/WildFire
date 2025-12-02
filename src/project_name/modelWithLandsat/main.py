import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time
import random
from dataset import PiedmontDataset
import segmentation_models_pytorch as smp
from typing import List,Dict
from model import MultiModalFPN
from train import train, val
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data.sampler import BatchSampler

# --- Includi la classe EarlyStopping qui o importala se è in un file separato ---
# Se l'hai messa in utils.py: from utils import EarlyStopping, HybridLoss
# Altrimenti, copia e incolla la classe EarlyStopping qui.
class EarlyStopping:
    """
    Ferma l'addestramento quando la validation metric non migliora dopo una data pazienza.
    """
    def __init__(self, patience=10, min_delta=0.001, mode='max', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.best_epoch = -1 

        if self.mode == 'min':
            self.best_metric = float('inf')
        elif self.mode == 'max':
            self.best_metric = float('-inf')
        else:
            raise ValueError("Mode must be 'min' or 'max'")

    def __call__(self, current_metric, epoch):
        if self.mode == 'max':
            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        elif self.mode == 'min':
            if current_metric < self.best_metric - self.min_delta:
                self.best_metric = current_metric
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_metric = float('inf')
        elif self.mode == 'max':
            self.best_metric = float('-inf')
        self.best_epoch = -1
# --- Fine classe EarlyStopping ---


###Config
DATA_ROOT = 'piedmont_new'
GEOJSON_PATH = 'piedmont_geojson/piedmont_2012_2024_fa.geojson'
LOG_DIR_BASE = './runsNew'
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TARGET_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_EPOCHS = 100 # Puoi mettere un numero molto alto qui, l'early stopping si fermerà prima
LEARNING_RATE_MAIN= 0.01
LEARNING_RATE_SENTINEL_ENCODER = 0.0001 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_CHANNELS_SENTINEL = 12
IN_CHANNELS_LANDSAT = 16
IN_CHANNELS_DEM = 1
IN_CHANNELS_STREETS = 1
IN_CHANNELS_ERA5_RASTER = 2
IN_CHANNELS_ERA5_TABULAR = 1

SWIR_BAND_INDEX = 9 
NIR_BAND_INDEX = 3 
RED_BAND_INDEX = 2 

###Not centered fires
EXCLUDE_DIRS = [
    'piedmont_new/fire_5292',
    'piedmont_new/fire_5295',
    'piedmont_new/fire_6069',
    'piedmont_new/fire_6654'
]


current_time = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(LOG_DIR_BASE, f"run_{current_time}")
writer = SummaryWriter(LOG_DIR)

experiment_notes = f"""
    Note Esperimento: {current_time}
    - Dataset: piedmont_new 
    - Architettura: MultiModalFPN (resnet50)
    - Split Dataset: 80% Train / 20% Validation
    - Loss Function: DiceLoss 
    - Ottimizzatore: SGD con LR {LEARNING_RATE_MAIN} e fine tuning sentinel ({LEARNING_RATE_SENTINEL_ENCODER}), weight decay 5e-4
    - Scheduler: CosineAnnealingLR T_MAX={NUM_EPOCHS}, CAMBIO MIN(5e-5)
    - Epoche: {NUM_EPOCHS} (con Early Stopping)
    - Batch Size: {BATCH_SIZE}
    - Input: 2 Pre-Fire Image (Sentinel-2: {IN_CHANNELS_SENTINEL} bands, 
      Landsat: {IN_CHANNELS_LANDSAT-1} bands e 1 flag band), 
      1 Other Data (Streets + DEM)
      1 Ignition Map (no Point)
      1 era5 (TAB + RASTER)
    - Augmentations: Spatial (flip, rotate, resize, crop, SHIFT).
    - Added use of CloudMask
    - Early Stopping: Patience=20 epoche, min_delta=0.001 sulla IoU di validazione.
    """
writer.add_text('Experiment_Notes', experiment_notes)

colors = ['black', 'red']
cmap_binary = ListedColormap(colors)

def denormalize_band_for_display(band_tensor: torch.Tensor, mean: float = None, std: float = None) -> np.ndarray:
    band_np = band_tensor.cpu().numpy()
    if mean is not None and std is not None:
        band_np = band_np * std + mean 
    max_val_for_display = 3000.0
    band_np = np.clip(band_np / max_val_for_display, 0, 1) 
    return band_np

def plot_prediction_vs_gt_and_input(writer, sentinel_image_tensor, landsat_image_tensor, prediction_tensor, gt_tensor, epoch, sample_name, threshold=0.5, global_stats=None):
    prediction_proba = torch.sigmoid(prediction_tensor).cpu().numpy()
    prediction_binary = (prediction_proba >= threshold).astype(np.uint8)
    gt_np = gt_tensor.cpu().numpy().astype(np.uint8)

    rgb_image = np.zeros((*sentinel_image_tensor.shape[1:], 3))

    if global_stats is not None:
        try:
            band_means = global_stats['mean_sentinel'].cpu().numpy()
            band_stds = global_stats['std_sentinel'].cpu().numpy()

            red_channel_display = denormalize_band_for_display(
                sentinel_image_tensor[SWIR_BAND_INDEX],
                mean=band_means[SWIR_BAND_INDEX],
                std=band_stds[SWIR_BAND_INDEX]
            )
            green_channel_display = denormalize_band_for_display(
                sentinel_image_tensor[NIR_BAND_INDEX],
                mean=band_means[NIR_BAND_INDEX],
                std=band_stds[NIR_BAND_INDEX]
            )
            blue_channel_display = denormalize_band_for_display(
                sentinel_image_tensor[RED_BAND_INDEX],
                mean=band_means[RED_BAND_INDEX],
                std=band_stds[RED_BAND_INDEX]
            )
            

            rgb_image = np.dstack((red_channel_display, green_channel_display, blue_channel_display))

        except IndexError as e:
            print(f"Warning: Band index out of range for {sample_name}. Check BAND_INDEX constants and IN_CHANNELS. Error: {e}")
            print(f"Input image tensor shape: {sentinel_image_tensor.shape}")
            rgb_image = np.zeros((*sentinel_image_tensor.shape[1:], 3))
        except Exception as e:
            print(f"Warning: Could not create RGB image for display for {sample_name}. Error: {e}")
            rgb_image = np.zeros((*sentinel_image_tensor.shape[1:], 3))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title(f"Input Sentinel (SWIR/NIR/RED, Epoch {epoch})")
    axes[0].axis('off')

    axes[1].imshow(gt_np, cmap=cmap_binary, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(prediction_binary, cmap=cmap_binary, vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (Epoch {epoch})")
    axes[2].axis('off')

    plt.tight_layout()

    writer.add_figure(f'Prediction_vs_GT_and_Input/{sample_name}_Epoch_{epoch}', fig, global_step=epoch)
    plt.close(fig)

def log_specific_samples(writer, model, dataset, sample_indices_to_log: list, epoch: int, tag_prefix: str, device: torch.device):
    model.eval()
    global_stats = getattr(dataset, 'global_stats', None)

    with torch.no_grad():
        for i, idx in enumerate(sample_indices_to_log):
            image_sentinel_tensor, image_landsat_tensor, dem_tensor, ignition_tensor,era5_raster,era5_tabular, landcover_tensor,gt_mask_tensor = dataset[idx]

            sentinel_img_for_plot = image_sentinel_tensor.cpu().clone()
            landsat_img_for_plot = image_landsat_tensor.cpu().clone()

            sentinel_tensor_for_model = image_sentinel_tensor.to(device).unsqueeze(0)
            landsat_tensor_for_model = image_landsat_tensor.to(device).unsqueeze(0)
            dem_tensor_for_model = dem_tensor.to(device).unsqueeze(0)
            ignition_tensor_for_model = ignition_tensor.to(device).unsqueeze(0)
            era5_tensor_for_model = era5_raster.to(device).unsqueeze(0)
            era5_tabular_for_model = era5_tabular.to(device).unsqueeze(0)
            
            output_logits, lc_logits= model(sentinel_tensor_for_model, landsat_tensor_for_model, 
                                  dem_tensor_for_model, ignition_tensor_for_model,era5_tensor_for_model,era5_tabular_for_model)

            plot_prediction_vs_gt_and_input(
                writer,
                sentinel_img_for_plot,
                landsat_img_for_plot, 
                output_logits.squeeze(0).squeeze(0), # Squeeze il batch e il canale (se num_classes=1)
                gt_mask_tensor.squeeze(0),
                epoch,
                f'{tag_prefix}_Sample_{idx}',
                global_stats=global_stats
            )
    model.train()


def main():
    print("Creating initial dataset instance to filter all valid directories and compute global statistics...")
    initial_full_dataset = PiedmontDataset(
        root_dir=DATA_ROOT,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=True,
        apply_augmentations=False, 
        global_stats=None, 
        initial_fire_dirs=None
    )
    
    global_stats = initial_full_dataset.global_stats 
    all_filtered_dirs = initial_full_dataset.fire_dirs 
    del initial_full_dataset
     
    all_filtered_dirs = [d for d in all_filtered_dirs if d not in EXCLUDE_DIRS]
    print(len(all_filtered_dirs), "valid fire directories found after excluding specified ones.")
    print("Global statistics computed and all valid directories identified.")
    
    random.seed(42)
    random.shuffle(all_filtered_dirs)

    total_dirs = len(all_filtered_dirs)
    train_dirs_size = int(0.8 * total_dirs)
    val_dirs_size = total_dirs - train_dirs_size

    train_dirs = all_filtered_dirs[:train_dirs_size]
    val_dirs = all_filtered_dirs[train_dirs_size:]

    print(f"Total fire directories after filtering: {total_dirs}")
    print(f"Train directories: {len(train_dirs)}")
    print(f"Validation directories: {len(val_dirs)}")

    train_dataset = PiedmontDataset(
        root_dir=DATA_ROOT,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=False, 
        apply_augmentations=True, 
        global_stats=global_stats, 
        initial_fire_dirs=train_dirs 
    )

    val_dataset = PiedmontDataset(
        root_dir=DATA_ROOT,
        geojson_path=GEOJSON_PATH,
        target_size=TARGET_SIZE,
        compute_stats=False, 
        apply_augmentations=False, 
        global_stats=global_stats, 
        initial_fire_dirs=val_dirs 
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
   
    model = MultiModalFPN(
        in_channels_sentinel=IN_CHANNELS_SENTINEL,
        in_channels_landsat=IN_CHANNELS_LANDSAT,
        in_channels_other_data=2,
        in_channels_ignition_map=1,
        in_channels_era5_raster=IN_CHANNELS_ERA5_RASTER,
        in_channels_era5_tabular=IN_CHANNELS_ERA5_TABULAR,
        num_classes=1
    ).to(DEVICE)
    
    #criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    
    burned_area_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    landcover_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE_MAIN, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4) 
    
    # Inizializza Early Stopping
    # Pazienza di 20 epoche, miglioramento minimo di 0.001 per l'IoU (modalità 'max')
    early_stopping = EarlyStopping(patience=30, min_delta=0.001, mode='max', verbose=True)
    
    best_iou = 0.0 

    num_fixed_samples_to_log = 3
    fixed_train_indices = random.sample(range(len(train_dataset)), num_fixed_samples_to_log)
    fixed_val_indices = random.sample(range(len(val_dataset)), num_fixed_samples_to_log)

    print(f"Fixed Train Sample Indices for logging: {fixed_train_indices}")
    print(f"Fixed Validation Sample Indices for logging: {fixed_val_indices}")

    w_mask = 0.5
    w_lc = 0.5

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss, train_ba_loss, train_lc_loss, train_iou = train(model, optimizer, train_loader, burned_area_loss_fn, landcover_loss_fn, DEVICE, writer, epoch, w_mask, w_lc)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Burned Area Loss: {train_ba_loss:.4f}, Landcover Loss: {train_lc_loss:.4f}, Train IoU: {train_iou:.4f}")
        val_loss, val_ba_loss, val_lc_loss, val_iou = val(model, val_loader, burned_area_loss_fn, landcover_loss_fn, DEVICE, writer, epoch)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Burned Area Loss: {val_ba_loss:.4f}, Landcover Loss: {val_lc_loss:.4f}, Validation IoU: {val_iou:.4f}")
        scheduler.step()

        writer.add_scalar('Metrics/Loss_Train', train_ba_loss, epoch)
        writer.add_scalar('Metrics/Loss_Validation', val_ba_loss, epoch)
        writer.add_scalar('Metrics/IoU_Train', train_iou, epoch)
        writer.add_scalar('Metrics/IoU_Validation', val_iou, epoch)
        writer.add_scalar('Learning_Rate/Current_LR', optimizer.param_groups[0]['lr'], epoch)

        logging_image_frequency = 5 
        if (epoch + 1) % logging_image_frequency == 0 or epoch == 0: 
            print(f"Logging {num_fixed_samples_to_log} fixed image samples at epoch {epoch+1}...")
            log_specific_samples(writer, model, train_dataset, fixed_train_indices, epoch, 'Train_Samples', DEVICE)
            log_specific_samples(writer, model, val_dataset, fixed_val_indices, epoch, 'Validation_Samples', DEVICE)

        # Gestione Early Stopping e Salvataggio del Modello Migliore
        # La logica di EarlyStopping controlla se fermare
        stop_training = early_stopping(val_iou, epoch) # Passiamo solo val_iou e l'epoca

        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            writer.add_text("Checkpoint", f"Model saved as best_model.pth at Epoch {epoch+1} with IoU: {best_iou:.4f}", global_step=epoch)
            
        # Se l'Early Stopping ha deciso di fermare, usciamo dal ciclo
        if stop_training:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            print(f"Best validation IoU was {early_stopping.best_metric:.4f} at epoch {early_stopping.best_epoch+1}.")
            break 

    writer.close()
    print("Training finished!")

if __name__ == '__main__':
    main()