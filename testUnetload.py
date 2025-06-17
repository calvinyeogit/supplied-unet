# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 11:06:31 2025
 
@author: mbiww
"""

########## TESTING WHAT VALUES MASK ARE
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image


# mask = Image.open("/Users/calvin/supplied-unet/dataset_wangwei/masks/fn-vcl-001_Zmap_mask.tif").convert("L")  # Convert to grayscale
# mask_np = np.array(mask)


# print("Min pixel value:", mask_np.min())
# print("Max pixel value:", mask_np.max())
# print("Unique pixel values:", np.unique(mask_np))
########## 
# Min pixel value: 0
# Max pixel value: 255
# Unique pixel values: [  0 255]
##########

import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from PyQt5 import QtGui, QtWidgets

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import matplotlib.pyplot as plt


## import dataset class, and load image stack
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from Unet_Config import Unet_Config
from dataset import dataset


from Losses import jaccard_distance_loss


def iou_score(preds: torch.Tensor, targets: torch.Tensor, eps=1e-6):
    """
    preds: (B,1,H,W) after sigmoid
    targets: (B,1,H,W) with values 0/1
    """
    preds_bin = (preds > 0.5).float()
    targets = targets.float()
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def display_images(image, cmap='gray',norm=None, interpolation='bilinear'):
#    
    plt.figure(figsize=(14,14))
    plt.axis('off')
    plt.imshow(image, cmap=cmap,
               norm=norm, interpolation=interpolation)
    plt.show()


## load image
test_dataset=dataset()
image_dir='/Users/calvin/supplied-unet/dataset_wangwei/images'
mask_dir='/Users/calvin/supplied-unet/dataset_wangwei/masks'

model_save_dir='/Users/calvin/supplied-unet/created_model'
model_name='testabcbig100.h5'


# image_dir = QtWidgets.QFileDialog.getExistingDirectory(caption="select the directory for traning images" )
# print('image dir is set to '+ image_dir)
# mask_dir=QtWidgets.QFileDialog.getExistingDirectory(caption="select the directory for traning masks" )
# print('mask dir is set to '+ mask_dir)

# model_save_dir=QtWidgets.QFileDialog.getExistingDirectory(caption="select the directory for traning models" )
# print('model saving dir is set to '+ model_save_dir)

model_name_full=model_save_dir+model_name


test_dataset.train_images=test_dataset.load_image(image_dir)
test_dataset.train_masks=test_dataset.load_image(mask_dir)

images=test_dataset.train_images
display_images(images[2])
display_images(test_dataset.train_masks[2])

## doing imaug
test_dataset.augment_images()
aug_img0=test_dataset.aug_images[5]
aug_mask0=test_dataset.aug_masks[5]
display_images(aug_img0[:,:,0])
display_images(aug_mask0[:,:,0])

def binary_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()


    
    
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from UNetModela import UNet

# Prepare data
image_np = np.stack(test_dataset.aug_images, axis=0).astype(np.float32)
mask_np = np.stack(test_dataset.aug_masks, axis=0).astype(np.float32)

image_np = np.squeeze(image_np, axis=-1)  # -> (N, 512, 512)
mask_np = np.squeeze(mask_np, axis=-1)

#image_tensor = torch.tensor(image_np).permute(0, 3, 1, 2)  # NHWC to NCHW
#mask_tensor = torch.tensor(mask_np).unsqueeze(1)           # NHW -> NCHW

image_tensor = torch.tensor(image_np).unsqueeze(1) 
mask_tensor = torch.tensor(mask_np).unsqueeze(1) 
# Dataset and DataLoader
dataset = TensorDataset(image_tensor, mask_tensor)
val_split = int(len(dataset) * 0.1)
train_set, val_set = random_split(dataset, [len(dataset)-val_split, val_split])
train_loader = DataLoader(
    train_set, 
    batch_size=2, 
    shuffle=True,
    num_workers=0,   # spawn zero worker processes in background due to Apple Chips
    pin_memory=True  # page-lock CPU tensors so transfers to MPS are faster
    )
val_loader = DataLoader(
    val_set, 
    batch_size=2,
    num_workers=0,   # spawn zero worker processes in background due to Apple Chips
    pin_memory=True  # page-lock CPU tensors so transfers to MPS are faster
    )

# Initialize model
model = UNet(in_ch=1, out_ch=1, sf=16)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)


# Optimizer and loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6, nesterov=True)
criterion = nn.BCELoss()

# Training loop
num_epochs_when_trialling = 5
num_epochs_when_real = 30

num_epochs = num_epochs_when_trialling



for epoch in range(num_epochs):
    
    t_start_epoch = time.time()
    
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_iou  = 0.0

    total_samples = 0

    for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
        
        # 1) Forward + backward + optimize
        imgs, masks = imgs.to(device), masks.to(device)
        batch_size = imgs.size(0)

        optimizer.zero_grad()
        outputs = model(imgs) # (B,1,H,W) raw logits
        loss = criterion(outputs, masks)
        acc = binary_accuracy(outputs.detach(), masks.detach())

        loss.backward()
        optimizer.step()

        # Free unused MPS memory if using MPS
        if device.type == "mps":
            torch.mps.empty_cache()
            
        # # 2) Compute IoU for this batch—no need for gradients
        # with torch.no_grad():
        #     probs = torch.sigmoid(outputs)       # (B,1,H,W) in [0,1]
        #     batch_iou = iou_score(probs, masks)  # your IoU‐helper returns a float
        #     running_iou += batch_iou * imgs.size(0)
        #     running_loss += loss.item() * imgs.size(0)

        # WW Suggestion from ChatGPT
        running_loss += loss.item() * batch_size
        running_acc += acc.item() * batch_size
        total_samples += batch_size
        
        # ---- Print progress per batch ----
        print(f"Epoch {epoch+1:2d} | Batch {batch_idx:3d}/{len(train_loader):3d}  "
              # f"  Batch IoU: {batch_iou:.4f}  Loss (this batch): {loss.item():.4f}"
              f"  Loss (this batch): {loss.item():.4f}")

        # running_loss += loss.item()
        # running_acc += acc.item()
        

    # epoch_loss = running_loss / len(train_loader) 

    # --------------- End of epoch ---------------
    epoch_iou  = running_iou  / len(train_loader)
    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    # epoch_time = time.time() - t_start_epoch
    print(f"running_acc: {running_acc}")

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.4f}")
    
    print(f"--- Epoch {epoch+1:2d} Complete: "
          f"Avg Loss: {epoch_loss:.4f}  Avg IoU: {epoch_iou:.4f}  "
          # f"Time: {epoch_time:.1f}s ---\n"
          )
    
    if device.type == "mps":
        torch.mps.empty_cache()


# Save model
torch.save(model.state_dict(), "unet_modeld.pth")
print("✔️ Training finished. Final model saved to unet_modeld.pth")

import os
import glob
from PIL import Image
import numpy as np

predict_dir = '/Users/calvin/supplied-unet/dataset_wangwei/forpredict'
output_dir = '/Users/calvin/supplied-unet/dataset_wangwei/predicted'
os.makedirs(output_dir, exist_ok=True) # Creates folder if missing (no error if exists)

# Load all image paths
image_paths = glob.glob(os.path.join(predict_dir, '*.tif')) + glob.glob(os.path.join(predict_dir, '*.jpg'))

print(f"Found {len(image_paths)} images in {predict_dir}")


def tile_image(img_np, tile_size=512):
    h, w = img_np.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    padded_img = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode='constant')
    tiles = []
    for i in range(0, padded_img.shape[0], tile_size):
        for j in range(0, padded_img.shape[1], tile_size):
            tile = padded_img[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
    return tiles, padded_img.shape[0] // tile_size, padded_img.shape[1] // tile_size, (pad_h, pad_w)


def untile_image(tiles, num_rows, num_cols, tile_size=512, padding=(0, 0)):
    full_image = np.zeros((num_rows * tile_size, num_cols * tile_size))
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            full_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tiles[idx]
            idx += 1
    if padding[0] > 0:
        full_image = full_image[:-padding[0], :]
    if padding[1] > 0:
        full_image = full_image[:, :-padding[1]]
    return full_image



#model = UNet(in_ch=1, out_ch=1).to(device)
## Load trained weights (from a .pth file)
#model.load_state_dict(torch.load("unet_modelc.pth", map_location=device))
#model.eval()  # Set model to evaluation mode


tile_size = 512

for idx, path in enumerate(image_paths):
    image = Image.open(path).convert('L')
    image_np = np.array(image)

    tiles, n_rows, n_cols, padding = tile_image(image_np, tile_size)
    tile_outputs = []

    for tile in tiles:
        tile_tensor = torch.tensor(tile).unsqueeze(0).unsqueeze(0).float().to(device) / 1.0
        with torch.no_grad():
            pred = model(tile_tensor)
        pred_np = pred.squeeze().cpu().numpy()
        #pred_np=pred_np*255
        pred_np = (pred_np > 0.3).astype(np.uint8) * 255
        tile_outputs.append(pred_np)

    full_pred = untile_image(tile_outputs, n_rows, n_cols, tile_size, padding)
    
    # Save prediction
    filename = os.path.splitext(os.path.basename(path))[0]
    outpath = os.path.join(output_dir, f"{filename}_predict.jpg")
    Image.fromarray(full_pred.astype(np.uint8)).save(outpath)

    print(f"Saved prediction: {outpath}")

