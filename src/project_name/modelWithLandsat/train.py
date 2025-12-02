import torch
import numpy as np
from utils import fast_hist, fire_area_iou, per_class_iou # Assicurati che utils.py sia disponibile
'''
def train(model, optimizer, dataloader, loss_fn, device, writer, epoch):
    model.train()
    hist = np.zeros((2, 2))  # Matrice di confusione 2x2: [TN, FP], [FN, TP]
    total_loss = 0.0
    
    for batch_idx, (image_sentinel,image_landsat,other_data,ignition_pt,era5_tensor, era5_tabular, gt_mask) in enumerate(dataloader):
        image_sentinel= image_sentinel.to(device)
        image_landsat= image_landsat.to(device)
        other_data =other_data.to(device)
        ignition_pt = ignition_pt.to(device)
        era5_tensor = era5_tensor.to(device)
        era5_tabular = era5_tabular.to(device)
        gt_mask = gt_mask.to(device)
        
        outputs = model(image_sentinel,image_landsat, other_data,ignition_pt, era5_tensor, era5_tabular) 
        
        loss = loss_fn(outputs, gt_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            predicted = (torch.sigmoid(outputs) > 0.5).float() 
            
            targets_flat = gt_mask.cpu().flatten().numpy().astype(int)
            predicted_flat = predicted.cpu().flatten().numpy().astype(int)
            
            hist += fast_hist(targets_flat, predicted_flat, 2)
        
    
    iou = fire_area_iou(hist)
    avg_loss = total_loss / len(dataloader)
    
    
    return avg_loss, iou

def val(model, dataloader, loss_fn, device, writer, epoch):
    
    model.eval() 
    hist = np.zeros((2, 2)) 
    total_loss = 0.0
    
    with torch.no_grad(): 
        for batch_idx, (image_sentinel,image_landsat,other_data, ignition_pt,era5_tensor, era5_tabular,gt_mask) in enumerate(dataloader):
            image_sentinel= image_sentinel.to(device)
            image_landsat= image_landsat.to(device)
            other_data =other_data.to(device)
            ignition_pt= ignition_pt.to(device)
            era5_tensor = era5_tensor.to(device)
            era5_tabular = era5_tabular.to(device)
            gt_mask = gt_mask.to(device)
            
            outputs = model(image_sentinel,image_landsat,other_data,ignition_pt,era5_tensor,era5_tabular)
            
            loss = loss_fn(outputs, gt_mask)
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            targets_flat = gt_mask.cpu().flatten().numpy().astype(int)
            predicted_flat = predicted.cpu().flatten().numpy().astype(int)
            
            hist += fast_hist(targets_flat, predicted_flat, 2)

    iou = fire_area_iou(hist)
    avg_loss = total_loss / len(dataloader)
    
    
    return avg_loss, iou
'''


def train(model, optimizer, dataloader, burned_area_loss_fn, landcover_loss_fn, device, writer, epoch, w_mask, w_landcover):
    model.train()
    hist_ba = np.zeros((2, 2))
    hist_lc = np.zeros((12, 12)) 
    total_loss = 0.0
    total_ba_loss = 0.0
    total_lc_loss = 0.0

    for batch_idx, (image_sentinel, image_landsat, other_data, ignition_pt, era5_tensor, era5_tabular, gt_landcover,gt_mask) in enumerate(dataloader):
        image_sentinel = image_sentinel.to(device)
        image_landsat = image_landsat.to(device)
        other_data = other_data.to(device)
        ignition_pt = ignition_pt.to(device)
        era5_tensor = era5_tensor.to(device)
        era5_tabular = era5_tabular.to(device)
        gt_mask = gt_mask.to(device)
        gt_landcover = gt_landcover.to(device).long()

        outputs_burned_area, outputs_landcover = model(image_sentinel, image_landsat, other_data, ignition_pt, era5_tensor, era5_tabular)
        loss_burned_area = burned_area_loss_fn(outputs_burned_area, gt_mask)
        #print("Landcover logits:", outputs_landcover.shape)
        #print("Landcover target:", gt_landcover.shape, gt_landcover.min().item(), gt_landcover.max().item())
        loss_landcover = landcover_loss_fn(outputs_landcover, gt_landcover.squeeze(1))
        loss = (w_mask * loss_burned_area) + (w_landcover * loss_landcover)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ba_loss += loss_burned_area.item()
        total_lc_loss += loss_landcover.item()

        with torch.no_grad():
            predicted_ba = (torch.sigmoid(outputs_burned_area) > 0.5).float()
            targets_ba_flat = gt_mask.cpu().flatten().numpy().astype(int)
            predicted_ba_flat = predicted_ba.cpu().flatten().numpy().astype(int)
            hist_ba += fast_hist(targets_ba_flat, predicted_ba_flat, 2)
            
            predicted_lc = torch.argmax(outputs_landcover, dim=1)
            targets_lc_flat = gt_landcover.cpu().flatten().numpy().astype(int)
            predicted_lc_flat = predicted_lc.cpu().flatten().numpy().astype(int)
            hist_lc += fast_hist(targets_lc_flat, predicted_lc_flat, 12)
            
    iou_ba = fire_area_iou(hist_ba)
    per_class_iou_scores = per_class_iou(hist_lc)

    # Per ottenere un singolo valore, puoi calcolare la media delle classi
    iou_lc= np.mean(per_class_iou_scores)
    
    avg_loss = total_loss / len(dataloader)
    avg_ba_loss = total_ba_loss / len(dataloader)
    avg_lc_loss = total_lc_loss / len(dataloader)

    return avg_loss, avg_ba_loss, avg_lc_loss, iou_ba

def val(model, dataloader, burned_area_loss_fn, landcover_loss_fn, device, writer, epoch):
    model.eval()
    
    hist_ba = np.zeros((2, 2))
    hist_lc = np.zeros((12, 12))
    
    total_loss = 0.0
    loss_ba_sum = 0.0
    loss_land_sum = 0.0
    
    with torch.no_grad():
        for batch_idx, (image_sentinel, image_landsat, other_data, ignition_pt, era5_raster, era5_tabular, gt_landcover, gt_mask) in enumerate(dataloader):
            image_sentinel = image_sentinel.to(device)
            image_landsat = image_landsat.to(device)
            other_data = other_data.to(device)
            ignition_pt = ignition_pt.to(device)
            era5_raster = era5_raster.to(device)
            era5_tabular = era5_tabular.to(device)
            gt_mask = gt_mask.to(device)
            gt_landcover = gt_landcover.to(device).long()
            
            outputs_burned_area, outputs_landcover = model(image_sentinel, image_landsat, other_data, ignition_pt, era5_raster, era5_tabular)
            
            loss_burned_area = burned_area_loss_fn(outputs_burned_area, gt_mask)
            loss_landcover = landcover_loss_fn(outputs_landcover, gt_landcover.squeeze(1))
            total_batch_loss = loss_burned_area + loss_landcover
            
            total_loss += total_batch_loss.item()
            loss_ba_sum += loss_burned_area.item()
            loss_land_sum += loss_landcover.item()
            
            predicted_ba = (torch.sigmoid(outputs_burned_area) > 0.5).float()
            targets_ba_flat = gt_mask.cpu().flatten().numpy().astype(int)
            predicted_ba_flat = predicted_ba.cpu().flatten().numpy().astype(int)
            hist_ba += fast_hist(targets_ba_flat, predicted_ba_flat, 2)
            
            predicted_lc = torch.argmax(outputs_landcover, dim=1)
            targets_lc_flat = gt_landcover.cpu().flatten().numpy().astype(int)
            predicted_lc_flat = predicted_lc.cpu().flatten().numpy().astype(int)
            hist_lc += fast_hist(targets_lc_flat, predicted_lc_flat, 12)
            
    iou_ba = fire_area_iou(hist_ba)
    per_class_iou_scores = per_class_iou(hist_lc)
    iou_lc = np.mean(per_class_iou_scores)
    # L'IoU del landcover viene calcolata, ma non restituita
    # iou_lc = multiclass_iou_from_hist(hist_lc) 
    
    avg_total_loss = total_loss / len(dataloader)
    avg_ba_loss = loss_ba_sum / len(dataloader)
    avg_land_loss = loss_land_sum / len(dataloader)
    
    return avg_total_loss, avg_ba_loss, avg_land_loss, iou_ba