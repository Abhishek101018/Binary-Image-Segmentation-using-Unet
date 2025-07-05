import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- U-Net model definition (must match your training script) ---
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

def preprocess_frame(frame, size=(256, 256)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    orig_img = img_pil.copy()
    img_pil = TF.resize(img_pil, size)
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # Shape: [1, 3, H, W]
    return img_tensor, orig_img

def postprocess_mask(mask_tensor, orig_size):
    mask = torch.sigmoid(mask_tensor)
    mask = (mask > 0.5).float()
    mask_np = mask.squeeze().cpu().numpy()
    # Ensure mask is 2D (H, W)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_img = mask_np * 255
    mask_img = Image.fromarray(mask_img.astype(np.uint8)).resize(orig_size, resample=Image.NEAREST)
    return mask_img

if __name__ == "__main__":
    model_path = "unet_checkpoint_epoch25.pth"
    video_path = "vid_6.mp4"  # Change to your video file

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    output_dir = "output_predicted_masked"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare VideoWriter for stacked output
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video for size.")
        exit(1)
    img_tensor, orig_img = preprocess_frame(frame)
    mask_img = postprocess_mask(torch.zeros_like(img_tensor), orig_img.size)
    mask_img_arr = np.array(mask_img)
    if mask_img_arr.ndim == 2:
        mask_img_rgb = np.stack([mask_img_arr]*3, axis=-1)
    else:
        mask_img_rgb = mask_img_arr
    input_np = np.array(orig_img)
    if mask_img_rgb.shape != input_np.shape:
        mask_img_rgb = cv2.resize(mask_img_rgb, (input_np.shape[1], input_np.shape[0]))
    max_display_width = 1600
    single_width = max_display_width // 2
    single_height = int(input_np.shape[0] * (single_width / input_np.shape[1]))
    frame_size = (single_width * 2, single_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stacked_video_path = os.path.join(output_dir, "stacked_output.mp4")
    stacked_writer = cv2.VideoWriter(stacked_video_path, fourcc, 20, frame_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_tensor, orig_img = preprocess_frame(frame)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
        mask_img = postprocess_mask(output, orig_img.size)

        # Stack input and predicted mask for real-time display
        mask_img_arr = np.array(mask_img)
        if mask_img_arr.ndim == 2:
            mask_img_rgb = np.stack([mask_img_arr]*3, axis=-1)
        else:
            mask_img_rgb = mask_img_arr
        input_np = np.array(orig_img)
        if mask_img_rgb.shape != input_np.shape:
            mask_img_rgb = cv2.resize(mask_img_rgb, (input_np.shape[1], input_np.shape[0]))
        input_np_disp = cv2.resize(input_np, (single_width, single_height))
        mask_img_rgb_disp = cv2.resize(mask_img_rgb, (single_width, single_height))
        combined = np.hstack([input_np_disp, mask_img_rgb_disp])

        # Show as video
        cv2.imshow("Input | Predicted Mask", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        stacked_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    stacked_writer.release()
    cv2.destroyAllWindows()
