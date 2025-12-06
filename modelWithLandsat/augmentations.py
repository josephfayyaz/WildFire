import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from typing import Tuple, Dict
import cv2

# Costanti per il numero di bande, utili per chiarezza e per evitare numeri magici
N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15 # Numero di bande Landsat "reali" (senza la banda di flag)
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1 # 15 + 1 = 16


class SentinelAugmentations:
    def __init__(self,
                 target_size: Tuple[int, int] = (256, 256),
                 p_flip: float = 0.6,
                 p_color: float = 0.3): 

        self.target_size = target_size

        self.spatial_transform = A.Compose([
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_flip),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.RandomResizedCrop(
                size=target_size,
                scale=(0.7, 0.9), #0.7,0.9
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.7, #0.5
            ),
        ], additional_targets={"landsat_image": "image",
                                "streets_image": "mask",
                                "dem_image": "image",
                                "era5_image": "image",
                                "ignition_pt": "mask",
                                "landcover_np": "mask"}) 


        self.to_tensor_transform = ToTensorV2()

    def __call__(self, image: np.ndarray, 
                 landsat_image: np.ndarray, 
                 streets_image: np.ndarray, 
                 dem_image: np.ndarray,
                 ignition_pt: np.ndarray,
                 era5_image: np.ndarray,
                 landcover_np: np.ndarray,
                 mask: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Applies transformation to Sentinel, Landsat, Street, dem, era5, ignition point and mask.
        
        Args:
            image (np.ndarray): Sentinel image (H, W, C).
            landsat_image (np.ndarray): Landsat image (H, W, C, inclusa la banda di flag).
            mask (np.ndarray): Ground truth mask (H, W).
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with transformed images and mask.
        """
        
        # --- Passaggio 1: Applicare le trasformazioni spaziali ---
        augmented_spatial = self.spatial_transform(image=image, 
                                                   landsat_image=landsat_image, 
                                                   streets_image=streets_image, 
                                                   dem_image=dem_image,
                                                   ignition_pt=ignition_pt,
                                                   era5_image=era5_image,
                                                   landcover_np=landcover_np,
                                                   mask=mask)

        augmented_sentinel_spatial = augmented_spatial['image']
        augmented_landsat_spatial = augmented_spatial['landsat_image'] 
        augmented_streets_spatial = augmented_spatial['streets_image']
        augmented_dem_spatial = augmented_spatial['dem_image']
        augmented_era5_spatial = augmented_spatial['era5_image']
        augmented_ignition_pt_spatial = augmented_spatial['ignition_pt']
        augmented_mask_spatial = augmented_spatial['mask']
        augmented_landcover_spatial = augmented_spatial['landcover_np']

        # Mask, streets, ignition pt don't have pixel level tarnsforms
        augmented_ignition_pt_final = augmented_ignition_pt_spatial
        augmented_streets_final = augmented_streets_spatial
        augmented_mask_final = augmented_mask_spatial

        # --- Passaggio 3: Conversione a Tensori PyTorch ---
        sentinel_image_tensor = self.to_tensor_transform(image=augmented_sentinel_spatial)['image']
        landsat_image_tensor = self.to_tensor_transform(image=augmented_landsat_spatial)['image']
        streets_tensor = self.to_tensor_transform(image= augmented_streets_final)['image']
        dem_tensor = self.to_tensor_transform(image=augmented_dem_spatial)['image']
        era5_tensor = self.to_tensor_transform(image=augmented_era5_spatial)['image']
        ignition_pt_tensor = self.to_tensor_transform(image=augmented_ignition_pt_final)['image']
        landcover_tensor = self.to_tensor_transform(image=augmented_landcover_spatial)['image']
        mask_tensor = self.to_tensor_transform(image=augmented_mask_final)['image']

        # Pulisci e formatta la maschera
        mask_tensor = mask_tensor.float() 
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1, :, :]

        # --- Passaggio 4: Restituisci un dizionario con tutti i tensori ---
        return {
            'sentinel_image': sentinel_image_tensor,
            'landsat_image': landsat_image_tensor,
            'streets_image': streets_tensor,
            'dem_image': dem_tensor,
            'ignition_pt': ignition_pt_tensor,
            'era5_image': era5_tensor,
            'landcover_np': landcover_tensor,
            'mask': mask_tensor
        }