import os
from typing import Tuple

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead

# Updated constants (kept identical to original for compatibility)
N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1

# DEM and streets have separate encoders in the full multimodal design
N_BANDS_DEM = 1
N_BANDS_STREETS = 1
N_BANDS_ERA5_RASTER = 2
N_BANDS_ERA5_TABULAR = 1
N_BANDS_IGNITION = 1


class MultiTaskFPN(smp.FPN):
    """
    Multi-task extension of SMP's FPN:
    - One head for burned-area segmentation (binary or multi-class)
    - One head for landcover segmentation
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights=None,
        in_channels: int = 3,
        burned_area_classes: int = 1,
        landcover_classes: int = 12,
        **kwargs,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=burned_area_classes,
            **kwargs,
        )

        self.burned_area_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=burned_area_classes,
            activation=None,
            upsampling=4,
            kernel_size=1,
        )

        self.landcover_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=landcover_classes,
            activation=None,
            upsampling=4,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          x: tensor of shape (B, C, H, W)

        Returns:
          burned_area_mask, landcover_mask
        """
        features = self.encoder(x)          # list/tuple of feature maps
        decoder_output = self.decoder(features)  # <-- key change here

        burned_area_mask = self.burned_area_head(decoder_output)
        landcover_mask = self.landcover_head(decoder_output)

        return burned_area_mask, landcover_mask

class AdvancedCrossModalFusionBlock(nn.Module):
    """
    Advanced fusion block for multi-modal feature maps.

    This class is kept for future multimodal extensions.
    The current Sentinel-only baseline does not use it,
    but it remains available as a building block.
    """

    def __init__(self, channel_dims: list, output_channels: int, reduction_ratio: int = 16):
        super().__init__()

        self.modalities = len(channel_dims)
        self.channel_dims = channel_dims
        total_channels = sum(channel_dims)

        # Channel attention to weight each channel
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, max(total_channels // reduction_ratio, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(total_channels // reduction_ratio, 1), total_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial attention to weight each spatial location
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(total_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final convolution to mix modalities into a common representation
        self.conv = nn.Conv2d(total_channels, output_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, modality_features: list) -> torch.Tensor:
        """
        Args:
            modality_features: list of feature maps [f1, f2, ..., fM],
                               each of shape (B, C_m, H, W)

        Returns:
            fused: tensor of shape (B, output_channels, H, W)
        """
        # Concatenate features along channel dimension
        concat_features = torch.cat(modality_features, dim=1)

        # Channel attention
        channel_weights = self.channel_attention(concat_features)
        # Spatial attention
        spatial_weights = self.spatial_attention(concat_features)

        # Apply both attentions
        attended = concat_features * channel_weights * spatial_weights

        # Final fusion conv
        fused = self.relu(self.bn(self.conv(attended)))
        return fused


class MultiModalFPN(nn.Module):
    """
    High-level multimodal model wrapper.

    Current baseline: Sentinel-only, but the signature and attributes
    are kept compatible with the original multimodal design so that
    additional modalities (Landsat, DEM, ERA5, ignition, etc.) can be
    added later without touching the training code.

    Forward signature (kept unchanged):

        def forward(
            self,
            sentinel_image: torch.Tensor,
            landsat_image: torch.Tensor,
            dem_image: torch.Tensor,
            ignition_map: torch.Tensor,
            era5_raster: torch.Tensor,
            era5_tabular,
        )

    Only `sentinel_image` is used in the baseline. The other arguments are
    accepted for API compatibility and can be integrated later.
    """

    def __init__(
        self,
        in_channels_sentinel: int = N_BANDS_SENTINEL,
        in_channels_landsat: int = N_BANDS_LANDSAT_WITH_FLAG,
        in_channels_other_data: int = N_BANDS_DEM + N_BANDS_STREETS + N_BANDS_IGNITION,
        in_channels_era5_raster: int = N_BANDS_ERA5_RASTER,
        in_channels_era5_tabular: int = N_BANDS_ERA5_TABULAR,
        in_channels_ignition_map: int = N_BANDS_IGNITION,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        landcover_classes: int = 12,
    ):
        super().__init__()

        # Core multi-task FPN using Sentinel bands as input
        self.base_fpn = MultiTaskFPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels_sentinel,
            burned_area_classes=num_classes,
            landcover_classes=landcover_classes,
        )

        # Expose commonly used attributes to preserve compatibility
        self.encoder = self.base_fpn.encoder
        self.decoder = self.base_fpn.decoder
        self.ba_segmentation_head = self.base_fpn.burned_area_head
        self.lc_segmentation_head = self.base_fpn.landcover_head

        # Store channel configs for future multimodal encoders (not used in baseline)
        self.in_channels_sentinel = in_channels_sentinel
        self.in_channels_landsat = in_channels_landsat
        self.in_channels_other_data = in_channels_other_data
        self.in_channels_era5_raster = in_channels_era5_raster
        self.in_channels_era5_tabular = in_channels_era5_tabular
        self.in_channels_ignition_map = in_channels_ignition_map

        # Determine the channel dimension of the deepest encoder feature (Sentinel branch)
        sentinel_deep_channels = self.encoder.out_channels[-1]
        self.sentinel_deep_channels = sentinel_deep_channels

        # Projection layers for each auxiliary modality. These layers map the input channels
        # of each modality to the same channel dimension as the deepest Sentinel feature map.
        self.landsat_proj = nn.Conv2d(
            self.in_channels_landsat, sentinel_deep_channels, kernel_size=1
        )
        self.other_proj = nn.Conv2d(
            self.in_channels_other_data, sentinel_deep_channels, kernel_size=1
        )
        self.era5_raster_proj = nn.Conv2d(
            self.in_channels_era5_raster, sentinel_deep_channels, kernel_size=1
        )

        # Project ERA5 tabular (scalar) to match spatial features using a small linear layer
        self.era5_tabular_proj = nn.Linear(
            self.in_channels_era5_tabular, sentinel_deep_channels
        )

        # Fusion block to combine Sentinel and auxiliary modalities at the deepest scale.
        # We fuse five modalities: Sentinel deep features, Landsat, other_data, ERA5 raster, and ERA5 tabular.
        self.fusion_block = AdvancedCrossModalFusionBlock(
            channel_dims=[sentinel_deep_channels] * 5,
            output_channels=sentinel_deep_channels,
        )

    def forward(
        self,
        sentinel_image: torch.Tensor,
        landsat_image: torch.Tensor,
        other_data: torch.Tensor,
        ignition_map: torch.Tensor,
        era5_raster: torch.Tensor,
        era5_tabular,
    ):
        """
        Forward pass for the multimodal model.

        Args:
            sentinel_image: Tensor of shape (B, C_sentinel, H, W)
            landsat_image: Tensor of shape (B, C_landsat, H, W)
            other_data: Tensor of shape (B, C_other, H, W) (DEM + streets + ignition)
            ignition_map: Unused dummy tensor kept for API compatibility
            era5_raster: Tensor of shape (B, C_era5_raster, H, W)
            era5_tabular: Tensor of shape (B, C_era5_tabular)

        Returns:
            burned_area_mask, landcover_mask
        """
        # Extract Sentinel encoder features
        features = self.encoder(sentinel_image)
        deep_feat = features[-1]
        target_h, target_w = deep_feat.shape[2], deep_feat.shape[3]

        # Project and pool Landsat
        landsat_feat = self.landsat_proj(landsat_image)
        landsat_feat = torch.nn.functional.adaptive_avg_pool2d(
            landsat_feat, (target_h, target_w)
        )

        # Project and pool other data (DEM + streets + ignition)
        other_feat = self.other_proj(other_data)
        other_feat = torch.nn.functional.adaptive_avg_pool2d(
            other_feat, (target_h, target_w)
        )

        # Project and pool ERA5 raster
        era5_feat = self.era5_raster_proj(era5_raster)
        era5_feat = torch.nn.functional.adaptive_avg_pool2d(
            era5_feat, (target_h, target_w)
        )

        # Project ERA5 tabular and expand spatially
        era5_tab_feat = self.era5_tabular_proj(era5_tabular)
        era5_tab_feat = era5_tab_feat.unsqueeze(-1).unsqueeze(-1)
        era5_tab_feat = era5_tab_feat.expand(
            -1, -1, target_h, target_w
        )

        # Fuse deepest features
        fused = self.fusion_block(
            [
                deep_feat,
                landsat_feat,
                other_feat,
                era5_feat,
                era5_tab_feat,
            ]
        )

        # Replace deepest Sentinel feature with fused feature
        fused_features = list(features)
        fused_features[-1] = fused

        # Decode and produce outputs
        decoder_output = self.decoder(fused_features)
        burned_area_mask = self.ba_segmentation_head(decoder_output)
        landcover_mask = self.lc_segmentation_head(decoder_output)

        return burned_area_mask, landcover_mask