# app.py — crack‑segmentation demo with brightness/contrast slider & example gallery
import os, cv2, torch, numpy as np
from PIL import Image
import gradio as gr
from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation
)

# ───────── CONFIG ───────────────────────────────────────────────────────
CKPT_PATH = "best_segformer_b0.pth"         # your fine‑tuned weights
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THR       = 0.50                            # probability threshold
IMG_SIZE  = 512                             # extractor resize

# ───────── LOAD MODEL & EXTRACTOR ──────────────────────────────────────
print("Loading model …")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

feature_extractor = SegformerFeatureExtractor(
    do_resize=True,
    size=IMG_SIZE,
    do_normalize=True,
    reduce_labels=False,
)

# ───────── OPTIONAL CONTRAST ENHANCEMENT ───────────────────────────────-

def enhance_contrast(rgb_uint8: np.ndarray, clip: float = 1.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement if clip>0 (approx range 0–3)."""
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
    """Return (original, mask, overlay) PIL images."""
    np_img = np.array(pil_img)

    # optional brightness/contrast enhancement for better visibility
    np_img_enh = enhance_contrast(np_img, clip=contrast)

    # model expects BGR; extractor handles resize / norm
    img_bgr = cv2.cvtColor(np_img_enh, cv2.COLOR_RGB2BGR)
    enc = feature_extractor(images=img_bgr, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits                  # [1,2,128,128]
        probs  = torch.softmax(logits, dim=1)[:, 1:2] # crack probability map
        preds  = (probs > THR).float()               # hard mask @128×128

        h, w = pil_img.height, pil_img.width
        mask_up = torch.nn.functional.interpolate(
            preds, size=(h, w), mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8)    # [H,W] {0,1}

    mask_pil = Image.fromarray(mask_up * 255)

    overlay = np_img_enh.copy()
    overlay[mask_up == 1] = [255, 0, 0]  # red overlay for crack
    overlay_pil = Image.fromarray(overlay)

    return Image.fromarray(np_img_enh), mask_pil, overlay_pil

# ───────── PRE‑LOADED EXAMPLE IMAGES  ───────────────────────────────────
EXAMPLES_DIR = "examples"  # put a few sample jpgs here
example_files = []
if os.path.isdir(EXAMPLES_DIR):
    example_files = [[os.path.join(EXAMPLES_DIR, f)]
                     for f in sorted(os.listdir(EXAMPLES_DIR))
                     if f.lower().endswith((".jpg", ".png"))][:8]

# ───────── GRADIO INTERFACE ────────────────────────────────────────────
IMAGE_OPTS = {"height": 256, "width": 256}

with gr.Blocks(title="Solar‑Cell Crack Segmentation (SegFormer‑B0)") as demo:
    gr.Markdown("## Solar‑Cell Crack Segmentation using SegFormer‑B0\nUpload an EL image or choose an example. Adjust **Contrast Enhance** if the image is too dark.")

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
