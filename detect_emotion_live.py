import cv2
import numpy as np
import tensorflow as tf
import time
import os

# config
MODEL_PATH = "emotion_model.h5"
OVERLAY_DIR = "overlays"
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
OVERLAY_SIZE = (128, 128)  # resize overlays to this size

# load model
print(f"🔍 Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# load emotion overlay
overlays = {}
for emotion in EMOTIONS:
    path = os.path.join(OVERLAY_DIR, f"{emotion}.png")
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, OVERLAY_SIZE)
        overlays[emotion] = img
        print(f"🖼️ Loaded overlay for '{emotion}'")
    else:
        print(f"⚠️ No overlay image found for '{emotion}'")

# camera handler
def open_camera(index=0):
    print("🎥 Attempting to open camera (make sure camera is not in use)...")
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    time.sleep(1)
    if not cap.isOpened():
        print("⚠️ MSMF backend failed, retrying with DirectShow...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        time.sleep(1)
    if not cap.isOpened():
        print("❌ ERROR: Could not access webcam.")
        return None
    print("✅ Webcam opened successfully!\n")
    return cap

cap = open_camera()
if cap is None:
    exit()

# function
def overlay_image_alpha(img, overlay, pos, alpha_mask):
    x, y = pos
    h, w = overlay.shape[0], overlay.shape[1]
    if x >= img.shape[1] or y >= img.shape[0]:
        return
    overlay_roi = img[y:y+h, x:x+w]
    alpha = alpha_mask[..., None]
    img[y:y+h, x:x+w] = alpha * overlay[..., :3] + (1 - alpha) * overlay_roi

# main loop
print("🤖 Real-time Emotion Detection with Overlay started! (Press 'q' to quit)")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Lost camera feed! Retrying...")
        cap.release()
        cap = open_camera()
        if cap is None:
            print("❌ Could not reconnect. Exiting...")
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        preds = model.predict(face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        confidence = np.max(preds)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # overlay emotion image
        if emotion in overlays:
            overlay_img = overlays[emotion]
            # get alpha channel or make
            if overlay_img.shape[2] == 4:
                alpha = overlay_img[:, :, 3] / 255.0
            else:
                alpha = np.ones(overlay_img.shape[:2], dtype=float)

            # position overlay over top right
            oy = max(y - OVERLAY_SIZE[1] - 10, 0)
            ox = min(x + w + 10, frame.shape[1] - OVERLAY_SIZE[0])
            overlay_image_alpha(frame, overlay_img[:, :, :3], (ox, oy), alpha)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
