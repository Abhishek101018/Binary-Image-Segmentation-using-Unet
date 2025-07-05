import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import random

# U-Net model (simple version)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

# Custom dataset
class RopeSegDataset(Dataset):
    def __init__(self, input_dir, mask_dir, transform=None):
        self.input_paths = []
        self.mask_paths = []
        self.transform = transform

        # Only use images that have a corresponding mask with the same name (case-insensitive, extension-insensitive)
        input_files = sorted(glob.glob(os.path.join(input_dir, '*')))
        mask_files = glob.glob(os.path.join(mask_dir, '*'))

        # Build a mapping from base name (without extension, lowercased) to mask path
        mask_map = {}
        for f in mask_files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            mask_map[base] = f

        for img_path in input_files:
            img_name = os.path.basename(img_path)
            img_base = os.path.splitext(img_name)[0].lower()
            if img_base in mask_map and os.path.exists(mask_map[img_base]):
                self.input_paths.append(img_path)
                self.mask_paths.append(mask_map[img_base])

        # Randomly shuffle the dataset (same order for images and masks)
        combined = list(zip(self.input_paths, self.mask_paths))
        random.shuffle(combined)
        if combined:
            self.input_paths, self.mask_paths = zip(*combined)
            self.input_paths = list(self.input_paths)
            self.mask_paths = list(self.mask_paths)
        else:
            self.input_paths = []
            self.mask_paths = []

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        img = Image.open(self.input_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

# Transform: resize, to tensor, normalize, and data augmentation
class JointTransform:
    def __init__(self, size=(256, 256), augment=True):
        self.size = size
        self.augment = augment

    def __call__(self, img, mask):
        img = TF.resize(img, self.size)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
            # Add more augmentations as needed

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()  # Ensure binary
        return img, mask

# Calculate accuracy
def calculate_accuracy(preds, masks):
    preds = (torch.sigmoid(preds) > 0.5).float()
    correct = (preds == masks).float().sum()
    total = torch.numel(preds)
    return correct / total

def dice_coefficient(preds, masks, epsilon=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    masks = (masks > 0.5).float()
    intersection = (preds * masks).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def predict_and_save(model, device, test_folder, out_folder, size=(256, 256)):
    os.makedirs(out_folder, exist_ok=True)
    model.eval()
    test_images = sorted(glob.glob(os.path.join(test_folder, '*')))
    for img_path in test_images:
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size
        img_resized = TF.resize(img, size)
        img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()
        mask_img = mask.squeeze().cpu().numpy() * 255
        mask_img = Image.fromarray(mask_img.astype(np.uint8)).resize(orig_size, resample=Image.NEAREST)
        out_path = os.path.join(out_folder, os.path.basename(img_path))
        mask_img.save(out_path)
        print(f"Saved predicted mask: {out_path}")

def train_unet(input_dir, mask_dir, test_dir, epochs=25, batch_size=8, lr=1e-3, checkpoint_path="unet_checkpoint.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use augmentation for training, not for validation
    train_transform = JointTransform(size=(256, 256), augment=True)
    val_transform = JointTransform(size=(256, 256), augment=False)

    # Load datasets (use training data for validation as well)
    train_dataset = RopeSegDataset(input_dir, mask_dir, transform=train_transform)
    val_dataset = RopeSegDataset(input_dir, mask_dir, transform=val_transform)

    # Check if dataset is empty
    if len(train_dataset) == 0:
        print("Error: No matching image/mask pairs found in the training set. Please check your input and output folders.")
        return
    if len(val_dataset) == 0:
        print("Error: No matching image/mask pairs found in the validation set. Please check your input and output folders.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize or load model
    model = UNet().to(device)
    if os.path.exists(checkpoint_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Training model from scratch...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # AdamW is often used for U-Net

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            # Ensure outputs and masks have the same shape: [B, 1, H, W]
            if outputs.dim() == 4 and outputs.shape[1] != 1:
                outputs = outputs.unsqueeze(1)
            if outputs.shape != masks.shape:
                outputs = outputs.view_as(masks)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks)

        train_dice /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                if outputs.dim() == 4 and outputs.shape[1] != 1:
                    outputs = outputs.unsqueeze(1)
                if outputs.shape != masks.shape:
                    outputs = outputs.view_as(masks)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(outputs, masks).item()
                val_dice += dice_coefficient(outputs, masks)

        val_accuracy /= len(val_loader)
        val_dice /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save(model.state_dict(), f"unet_checkpoint_epoch{epoch+1}.pth")
            print(f"Checkpoint saved at unet_checkpoint_epoch{epoch+1}.pth")

    print("Training complete.")

    # Predict on test images and save results
    predict_and_save(model, device, test_dir, test_dir, size=(256, 256))

if __name__ == "__main__":
    train_unet("input", "output", "test")
