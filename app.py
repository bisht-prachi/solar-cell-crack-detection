# app.py — crack segmentation demo with CLAHE + Gradio for smp.Unet
import os, cv2, torch, numpy as np
from PIL import Image
import gradio as gr
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# ───────── CONFIG ───────────────────────────────────────────────────────
CKPT_PATH = "best_model.pth"         # your fine‑tuned weights
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THR       = 0.50                      # probability threshold
IMG_SIZE  = 448                       # input size to match training

# ───────── LOAD U‑NET MODEL ────────────────────────────────────────────
print("Loading model …")
model = smp.Unet(
    encoder_name="resnet34",        # or mobilenet_v2 etc.
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ───────── TRANSFORM (MATCH TRAINING NORMALIZATION) ─────────────────────
transform = T.Compose([
    T.ToTensor(),                               # [H,W,C] → [C,H,W]
    T.Resize((IMG_SIZE, IMG_SIZE)),             # resize for model
    T.Normalize(mean=[0.485, 0.456, 0.406],      # ImageNet stats
                std=[0.229, 0.224, 0.225])
])

# ───────── CONTRAST ENHANCEMENT ─────────────────────────────────────────
def enhance_contrast(rgb_uint8: np.ndarray, clip: float = 1.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement if clip > 0 (approx 0–3)."""
    if clip <= 0.01:
        return rgb_uint8
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# ───────── INFERENCE FUNCTION ───────────────────────────────────────────
def predict_mask(pil_img: Image.Image, contrast: float = 0.0):
    """Return (enhanced original, predicted mask, overlay) PIL images."""
    np_img = np.array(pil_img)
    np_img_enh = enhance_contrast(np_img, clip=contrast)

    # prepare model input
    img_tensor = transform(Image.fromarray(np_img_enh)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)                # [1,1,H,W]
        probs = torch.sigmoid(logits)
        preds = (probs > THR).float()             # binary mask

        # upsample mask to original image size
        preds_up = torch.nn.functional.interpolate(
            preds, size=pil_img.size[::-1], mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8)

    mask_pil = Image.fromarray(preds_up * 255)

    # overlay
    overlay = np_img_enh.copy()
    overlay[preds_up == 1] = [255, 0, 0]
    overlay_pil = Image.fromarray(overlay)

    return Image.fromarray(np_img_enh), mask_pil, overlay_pil

# ───────── PRE‑LOADED EXAMPLES ──────────────────────────────────────────
EXAMPLES_DIR = "examples"
example_files = []
if os.path.isdir(EXAMPLES_DIR):
    example_files = [[os.path.join(EXAMPLES_DIR, f)]
                     for f in sorted(os.listdir(EXAMPLES_DIR))
                     if f.lower().endswith((".jpg", ".png"))][:8]

# ───────── GRADIO APP ───────────────────────────────────────────────────
IMAGE_OPTS = {"height": 256, "width": 256}

with gr.Blocks(title="Solar‑Cell Crack Segmentation (U‑Net)") as demo:
    gr.Markdown("## Solar‑Cell Crack Segmentation using U‑Net\nUpload an EL image or choose an example. Adjust **Contrast Enhance** if the image is too dark.")

    with gr.Row():
        inp_img = gr.Image(type="pil", label="Input EL image")
        slider  = gr.Slider(minimum=0.0, maximum=3.0, step=0.1, value=0.0,
                            label="Contrast Enhance (CLAHE clipLimit)")

    with gr.Row():
        out_orig   = gr.Image(type="pil", label="Enhanced Original", **IMAGE_OPTS)
        out_mask   = gr.Image(type="pil", label="Predicted Mask", **IMAGE_OPTS)
        out_overlay= gr.Image(type="pil", label="Overlay", **IMAGE_OPTS)

    btn = gr.Button("Segment Cracks")
    btn.click(fn=predict_mask, inputs=[inp_img, slider], outputs=[out_orig, out_mask, out_overlay])

    if example_files:
        gr.Examples(examples=example_files, inputs=[inp_img], label="Example images")

if __name__ == "__main__":
    demo.launch()
