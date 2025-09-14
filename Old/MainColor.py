import cv2
import pandas as pd
import numpy as np
import sys
import time
import argparse
from threading import Thread

# ---------- Settings / CLI ----------
parser = argparse.ArgumentParser(description="IP camera color detection (threaded & optimized)")
parser.add_argument("--url", type=str, default="", help="Camera URL (rtsp:// or http://). If empty, uses local camera index 0")
parser.add_argument("--csv", type=str, default="colornames.csv", help="Path to color names CSV (with hex and name columns)")
parser.add_argument("--min_area", type=int, default=1000, help="Minimum contour area to consider")
parser.add_argument("--sat_thresh", type=int, default=80, help="Min saturation to accept pixel")
parser.add_argument("--val_thresh", type=int, default=80, help="Min brightness (value) to accept pixel")
args = parser.parse_args()

CAMERA_URL = args.url
CSV_PATH = args.csv
MIN_CONTOUR_AREA = args.min_area
SAT_THRESH = args.sat_thresh
VAL_THRESH = args.val_thresh

# ---------- Load colors CSV ----------
def hex_to_rgb(hex_str):
    hex_str = str(hex_str).lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

try:
    colors_df = pd.read_csv(CSV_PATH)
except Exception as e:
    print("Failed to load CSV:", e)
    sys.exit(1)

if "hex" not in colors_df.columns or "name" not in colors_df.columns:
    print("CSV must contain 'hex' and 'name' columns.")
    sys.exit(1)

colors_df[["R", "G", "B"]] = colors_df["hex"].apply(lambda x: pd.Series(hex_to_rgb(x)))

def get_color_name(b, g, r):
    # OpenCV gives BGR → convert to RGB for comparison
    R, G, B = r, g, b
    min_dist = 1e9
    cname, hex_code = "Unknown", "#000000"
    for i in range(len(colors_df)):
        dr, dg, db = int(colors_df.loc[i, "R"]), int(colors_df.loc[i, "G"]), int(colors_df.loc[i, "B"])
        dist = abs(R - dr) + abs(G - dg) + abs(B - db)
        if dist < min_dist:
            min_dist = dist
            cname = colors_df.loc[i, "name"]
            hex_code = colors_df.loc[i, "hex"]
    return cname, hex_code

# ---------- Threaded Camera Class ----------
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url if url else 0, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print("Failed to open camera:", url)
            sys.exit(1)
        self.frame = None
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ---------- Main ----------
stream = CameraStream(CAMERA_URL)
time.sleep(1)
print("Press 'q' to quit. 's' to save snapshot.jpg")

while True:
    frame = stream.read()
    if frame is None:
        time.sleep(0.01)
        continue

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1].astype(np.int32), hsv[:, :, 2].astype(np.int32)
        mask = (sat >= SAT_THRESH) & (val >= VAL_THRESH)

        if np.any(mask):
            idx_flat = np.argmax((sat * mask) * 1000 + val * mask)
            idx = np.unravel_index(idx_flat, sat.shape)
            py, px = int(idx[0]), int(idx[1])
            b, g, r = roi[py, px].tolist()
        else:
            cy, cx = h // 2, w // 2
            b, g, r = roi[cy, cx].tolist()

        color_name, hex_code = get_color_name(b, g, r)

        # Convert hex to BGR for drawing
        draw_color = tuple(int(hex_code.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
        label = f"{color_name} {hex_code}"
        cv2.rectangle(frame, (x, y - 26), (x + 220, y), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("IP Color Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("snapshot.jpg", frame)
        print("Saved snapshot.jpg")

stream.stop()
cv2.destroyAllWindows()