import cv2
import os

OVERLAY_DIR = "overlays"
TARGET_SIZE = (128, 128)

for filename in os.listdir(OVERLAY_DIR):
    if not filename.endswith(".png"):
        continue
    path = os.path.join(OVERLAY_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Could not read {filename}")
        continue

    resized = cv2.resize(img, TARGET_SIZE)
    save_path = os.path.join(OVERLAY_DIR, filename)
    cv2.imwrite(save_path, resized)
    print(f"✅ Resized {filename} → {TARGET_SIZE}")
