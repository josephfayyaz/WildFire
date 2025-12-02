import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torchgeo.models.resnet import ResNet50_Weights
import os
from segmentation_models_pytorch.base import SegmentationHead

# Aggiornamento delle costanti
N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1
# Ora le strade e il DEM hanno encoder separati
N_BANDS_DEM = 1
N_BANDS_STREETS = 1
N_BANDS_ERA5_RASTER = 2
N_BANDS_ERA5_TABULAR = 1
N_BANDS_IGNITION = 1


class MultiTaskFPN(smp.FPN):
    def __init__(self, 
                 encoder_name: str = "resnet50",
                 encoder_weights = None,
                 in_channels: int = 3,
                 burned_area_classes: int = 1,
                 landcover_classes: int = 12,
                 **kwargs):
        
        # Inizializza la FPN base con le classi burned area
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=burned_area_classes,
            **kwargs
        )
        
        # La segmentation head originale diventa quella per burned area
        self.burned_area_head = self.segmentation_head
        
        # Crea una nuova segmentation head per landcover
        self.landcover_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=landcover_classes,
            activation=None,
            upsampling=4,
            kernel_size=1
        )
    
    def forward(self, x):
        """Override del forward per restituire entrambi gli output"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        
        burned_area_mask = self.burned_area_head(decoder_output)
        landcover_mask = self.landcover_head(decoder_output)
        
        return burned_area_mask, landcover_mask
 
# Blocco di fusione avanzato
class AdvancedCrossModalFusionBlock(nn.Module):
    def __init__(self, channel_dims: list, output_channels: int, reduction_ratio=16):
        super().__init__()
        
        self.modalities = len(channel_dims)
        self.channel_dims = channel_dims
        total_channels = sum(channel_dims)

        # Attenzione di canale per assegnare un peso a ogni canale
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // reduction_ratio, total_channels, 1),
            nn.Sigmoid()
        )
        
        # Attenzione spaziale per assegnare un peso a ogni pixel
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(total_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Fusione finale con una convoluzione
        self.conv = nn.Conv2d(total_channels, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, modality_features):
        concat_features = torch.cat(modality_features, dim=1)
        
        # Calcola e applica l'attenzione di canale e spaziale
        channel_weights = self.channel_attention(concat_features)
        spatial_weights = self.spatial_attention(concat_features)
        
        attended_features = concat_features * channel_weights * spatial_weights
        
        fused = self.relu(self.bn(self.conv(attended_features)))
        
        return fused

# Sostituzione della classe principale
class MultiModalFPN(nn.Module):
    def __init__(self, 
                 in_channels_sentinel: int = N_BANDS_SENTINEL, 
                 in_channels_landsat: int = N_BANDS_LANDSAT_WITH_FLAG,
                 in_channels_other_data: int = N_BANDS_DEM + N_BANDS_STREETS, # Nuovo
                 in_channels_era5_raster: int = N_BANDS_ERA5_RASTER,
                 in_channels_era5_tabular: int = N_BANDS_ERA5_TABULAR,
                 in_channels_ignition_map: int = 1,
                 num_classes: int = 1,
                 encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet"):
        super().__init__()

        # --- 1. Encoder Sentinel-2 (invariato) ---
        self.sentinel_encoder = smp.FPN(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=in_channels_sentinel,
            classes=1
        ).encoder
        
        # --- 2. Landsat Encoder ---
        self.landsat_encoder = smp.FPN(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels_landsat, 
            classes=1,
            encoder_depth=5
        ).encoder
        
        # --- 3. Dem + Streets Encoder
        self.dem_encoder = smp.FPN(
            encoder_name="resnet18", 
            encoder_weights=encoder_weights,
            in_channels=in_channels_other_data,
            classes=1
        ).encoder

        # --- 4. Ignition Map Encoder ---
        self.ignition_map_processor = smp.FPN(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels_ignition_map,
            classes=1
        ).encoder

        
        # --- 5. ERA5 Encoder ---
        self.era5_raster_encoder = smp.FPN(
            encoder_name="resnet18", 
            encoder_weights=None, 
            in_channels=in_channels_era5_raster,
            classes=1
        ).encoder

        # MLP ERA 5 Tabular (Temperature and Pressure)
        self.era5_tabular_feature_dim = 16 
        self.era5_tabular_processor = nn.Sequential(
            nn.Linear(in_channels_era5_tabular, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.era5_tabular_feature_dim),
            nn.ReLU(inplace=True)
        )
        '''
        base_unet = smp.FPN(
            encoder_name="resnet50",
            encoder_weights=None, # Il decoder non usa pesi pre-allenati direttamente
            in_channels=in_channels_sentinel, # Definisce la struttura "base" del decoder
            classes=num_classes,
            activation=None # Mantieni l'output lineare per la loss function (es. BCEWithLogitsLoss)
        )
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head
        '''
        multi_task_fpn = MultiTaskFPN(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=in_channels_sentinel,
            burned_area_classes=num_classes,
            landcover_classes=12, # Numero di classi per la landcover
            activation=None
        )
        self.decoder = multi_task_fpn.decoder
        self.ba_segmentation_head = multi_task_fpn.burned_area_head
        self.lc_segmentation_head = multi_task_fpn.landcover_head
        '''
        # --- 5. Fusion Blocks 
        self.fusion_blocks = nn.ModuleList()
        num_encoder_stages = len(self.sentinel_encoder.out_channels) # Numero di stadi dell'encoder (es. 6 per resnet34)
        
        for i in range(num_encoder_stages):
            sentinel_out_c = self.sentinel_encoder.out_channels[i]
            landsat_out_c = self.landsat_encoder.out_channels[i]
            dem_out_c = self.dem_encoder.out_channels[i]
            ignition_out_c = self.ignition_map_processor.out_channels[i] # Canali dall'encoder ignition
            era5_raster_out_c = self.era5_raster_encoder.out_channels[i]

            # The total input channels to the fusion block will be the SUM of the channels from all 4 encoders.
            fused_in_channels = (
                sentinel_out_c + 
                landsat_out_c + 
                dem_out_c + 
                ignition_out_c + 
                era5_raster_out_c + # NUOVO
                self.era5_tabular_feature_dim # NUOVO: i canali dell'output MLP espanso
            )
            
            # The output of the fusion block must match the channels that the decoder expects.
            # As the decoder has been configured based on the Sentinel encoder,
            # the desired output channels for fusion at this stage are those of Sentinel.
            fusion_output_channels = sentinel_out_c

            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(fused_in_channels, fusion_output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(fusion_output_channels),
                    nn.ReLU(inplace=True)
                )
            )
        '''
        # --- 7. Aggiornamento dei Fusion Blocks
        self.fusion_blocks = nn.ModuleList()
        num_encoder_stages = len(self.sentinel_encoder.out_channels)
        
        for i in range(num_encoder_stages):
            # Lista dei canali di output per ogni encoder in questo stadio
            channel_dims_for_stage = [
                self.sentinel_encoder.out_channels[i],
                self.landsat_encoder.out_channels[i],
                self.dem_encoder.out_channels[i], # Nuovo
                self.ignition_map_processor.out_channels[i],
                self.era5_raster_encoder.out_channels[i]
            ]
            if i == num_encoder_stages - 1: # The deepest stage
                # Add the tabular feature dimension for the deepest block
                channel_dims_for_stage.append(self.era5_tabular_feature_dim)
            
            fusion_output_channels = self.sentinel_encoder.out_channels[i]
            
            self.fusion_blocks.append(
                AdvancedCrossModalFusionBlock(
                    channel_dims=channel_dims_for_stage,
                    output_channels=fusion_output_channels
                )
            )
        
    def forward(self, 
                sentinel_image: torch.Tensor, 
                landsat_image: torch.Tensor, 
                dem_image: torch.Tensor, # Nuovo input
                ignition_map: torch.Tensor,
                era5_raster: torch.Tensor,
                era5_tabular) -> torch.Tensor:
        
        # ---Estract feature from every encoder ---
        sentinel_features = self.sentinel_encoder(sentinel_image)
        landsat_features = self.landsat_encoder(landsat_image)
        dem_features = self.dem_encoder(dem_image) 
        ignition_features = self.ignition_map_processor(ignition_map)
        era5_raster_features = self.era5_raster_encoder(era5_raster)

        # --- Estract flag Landsat ---
        landsat_presence_flag_input = landsat_image[:, N_BANDS_LANDSAT_ACTUAL, :, :]
        landsat_presence_flag_input = landsat_presence_flag_input.unsqueeze(1)
        
        era5_tabular_processed = self.era5_tabular_processor(era5_tabular)
        
        # --- Step 3: Fuse features at each level (stage) of the encoder (Modificato per includere il nuovo encoder) ---
        fused_features = []
        num_encoder_stages = len(self.sentinel_encoder.out_channels) ##mod
        # Ora facciamo lo zip su TUTTE E TRE le liste di feature.
        for i, (sentinel_feat, landsat_feat, dem_feat,ignition_feat,era5_raster_feat) in enumerate(zip(sentinel_features, landsat_features,dem_features,ignition_features,era5_raster_features)):
            # Resize Landsat presence flag.
            landsat_flag_resized = F.interpolate(
                landsat_presence_flag_input,
                size=landsat_feat.shape[2:], 
                mode='nearest', 
                align_corners=None 
            )
            
            # Apply flag to Landsat features.
            masked_l_feat = landsat_feat * landsat_flag_resized 

             # --- NUOVO: Espansione spaziale delle feature ERA5 tabulari ---
            # Ottieni le dimensioni spaziali dell'attuale stadio di feature per l'espansione
            h, w = sentinel_feat.shape[2:] 
            expanded_era5_tabular_feat = era5_tabular_processed.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)

            '''# We concatenate Sentinel, Landsat masked features, features from 'other_data_encoder' and era5 features
            concatenated_features = torch.cat([sentinel_feat, masked_l_feat, dem_feat,ignition_feat,era5_raster_feat,expanded_era5_tabular_feat], dim=1) 
            
            # Apply the merge block
            adapted_features = self.fusion_blocks[i](concatenated_features)
            fused_features.append(adapted_features)
            '''
            # Crea la lista di tensori per la concatenazione
            modality_features_list = [
                sentinel_feat, 
                masked_l_feat, 
                dem_feat,
                ignition_feat,
                era5_raster_feat
            ]
            
            # Aggiungi le feature tabulari di ERA5 solo all'ultimo stadio, come previsto dall'init
            if i == num_encoder_stages - 1:
                modality_features_list.append(expanded_era5_tabular_feat)
            adapted_features = self.fusion_blocks[i](modality_features_list)
            fused_features.append(adapted_features)
        # --- Step 4: Pass the fused features to the shared decoder (Nessuna modifica qui) ---
        decoder_output = self.decoder(fused_features)

        # --- Step 5: Pass the decoder's output to the segmentation head (Nessuna modifica qui) ---
        #final_mask = self.segmentation_head(decoder_output)
        final_mask, landcover_mask = self.ba_segmentation_head(decoder_output), self.lc_segmentation_head(decoder_output)

        return final_mask, landcover_mask