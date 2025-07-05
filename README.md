# Rope Segmentation using U-Net

This repository contains a full training pipeline for binary image segmentation using a U-Net architecture implemented in PyTorch. The application demonstrated here is for segmenting ropes or similar elongated structures from RGB images using corresponding binary masks.

---

## 🧠 What is U-Net?

U-Net is a type of Convolutional Neural Network (CNN) originally designed for biomedical image segmentation. It was introduced in the 2015 paper *"U-Net: Convolutional Networks for Biomedical Image Segmentation"* by Ronneberger et al.

**Key Characteristics:**

- **Encoder-Decoder structure:** U-Net has a symmetric encoder-decoder architecture.
- **Skip Connections:** It copies feature maps from the encoder and concatenates them with the decoder to retain spatial information.
- **Output:** Produces a dense pixel-wise prediction, making it ideal for segmentation tasks.

U-Net is popular because it can work with fewer training images and yields more precise segmentations, especially for small and thin structures.

---

## 🏗 U-Net Architecture (Implemented Here)

```
Input Image
   |
Encoder (Downsampling):
   - Conv2d -> ReLU -> BatchNorm (x2)
   - MaxPooling (x2)
   |
Bottleneck:
   - Deeper Conv layers
   |
Decoder (Upsampling):
   - ConvTranspose2d (x2)
   - Skip connections from encoder
   - Conv2d -> ReLU -> BatchNorm (x2)
   |
Final Conv Layer
   |
Segmentation Mask Output (1 channel, Sigmoid activated)
```

---

## 📁 Directory Structure

```
.
├── train_unet.py              # Main script to train and evaluate the model
├── input/                     # Folder with training images
├── output/                    # Folder with corresponding binary masks
├── test/                      # Folder containing test images (no masks needed)
├── unet_checkpoint_epochX.pth # Saved model checkpoints
├── output_predicted_masked/   # Folder where predicted masks are saved
└── README.md                  # Project documentation
```

---

## 🚶 Workflow Overview

### 1. **Custom Dataset Class**

- Loads image-mask pairs from `input/` and `output/` folders.
- Matches filenames (case-insensitive) and verifies pair integrity.
- Applies data augmentations like flipping and rotation.

### 2. **U-Net Model Initialization**

- A simplified version of U-Net is created with three encoding and two decoding layers.
- Final output layer maps to a 1-channel mask (binary).

### 3. **Training Loop**

- Uses Binary Cross-Entropy with Logits loss (`BCEWithLogitsLoss`).
- Uses **AdamW** optimizer.
- Computes **Dice Coefficient** for measuring segmentation performance.
- Every 5 epochs, a checkpoint is saved.

### 4. **Evaluation on Test Data**

- All images in `test/` folder are passed through the trained model.
- Predicted binary masks are generated and saved side-by-side.

---

## ⚙️ How to Use

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pillow opencv-python
```

### 2. Prepare Dataset

- Place your training RGB images in the `input/` directory.
- Place your binary mask images (same name) in the `output/` directory.
- Place any test images in the `test/` directory.

### 3. Train the Model

```bash
python train_unet.py
```

Optional arguments to modify training:

```python
train_unet("input", "output", "test", epochs=25, batch_size=8, lr=1e-3)
```

### 4. Output

- After training, predictions for all test images are saved in the `test/` folder.
- Checkpoints are saved every 5 epochs.

---

## 📈 Metrics

- **Loss:** Binary Cross Entropy
- **Accuracy:** Pixel-wise accuracy
- **Dice Coefficient:** Overlap metric between predicted and true masks

---

## 🖼 Example Results

After training, each image in the `test/` folder will produce a corresponding predicted mask with:

- White for rope (foreground)
- Black for background

---

## 📌 Notes

- Input/output masks are resized to 256x256 during training and testing.
- All augmentations are applied only to training data.
- Masks are binarized to ensure correct prediction.

---

## 🧪 Extensions (Ideas)

- Use larger/deeper U-Net
- Add more augmentations: color jitter, blur, etc.
- Multi-class segmentation
- Evaluation with IoU (Intersection over Union)

---

## 📜 License

This project is open-source and free to use under the MIT License.

---

## 👨‍💻 Author

Created by **Abhi** using PyTorch.

If you use this project, a star on the GitHub repo would be appreciated! ⭐

