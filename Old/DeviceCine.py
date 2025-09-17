import cv2
import pandas as pd
import numpy as np
import sys
import threading
import os
import msvcrt  # Windows keyboard input

# ---------- Settings ----------
CSV_PATH = "colornames.csv"
MIN_CONTOUR_AREA = 1000
SAT_THRESH = 80
VAL_THRESH = 80

# Paths where images will be saved (Tank folders)
SAVE_DIRS = [f"captures/tank{i+1}" for i in range(3)]  # 3 tanks

# Create directories if not exist
for d in SAVE_DIRS:
    os.makedirs(d, exist_ok=True)

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

colors_df[["R", "G", "B"]] = colors_df["hex"].apply(
    lambda x: pd.Series(hex_to_rgb(x))
)

def get_color_name(b, g, r):
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

# ---------- CameraStream ----------
class CameraStream:
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.frame = None
        self.stopped = False
        self.index = cam_index
        self.counter = 0  # for unique filenames

        if not self.cap.isOpened():
            self.cap = None
            return

        print(f"✅ Using camera index {cam_index}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def save_frame(self):
        """Save current frame into its own Tank directory with unique filename"""
        if self.frame is None:
            return None
        save_dir = SAVE_DIRS[self.index % len(SAVE_DIRS)]
        self.counter += 1
        filename = os.path.join(save_dir, f"capture_{self.counter}.jpg")
        cv2.imwrite(filename, self.frame)
        return filename

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# ---------- Processing ----------
def detect_colors(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_colors = []

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
        detected_colors.append((color_name, hex_code))

        draw_color = tuple(int(hex_code.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
        label = f"{color_name} {hex_code}"
        cv2.rectangle(frame, (x, y - 26), (x + 220, y), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, detected_colors

# ---------- Multi-camera Main ----------
def run_multicamera(max_cams=3):
    streams = []
    for i in range(max_cams):
        stream = CameraStream(i)
        if stream.cap:
            streams.append(stream)

    if not streams:
        print("❌ No working cameras found!")
        sys.exit(1)

    print("Type 'click' (then Enter) to capture from all tanks, 'quit' to exit.\n")

    buffer = ""

    while True:
        for stream in streams:
            frame = stream.read()
            if frame is not None:
                window_name = f"Tank {stream.index+1}"
                cv2.imshow(window_name, cv2.resize(frame, (480, 360)))

        # Check keyboard input (Windows safe)
        if msvcrt.kbhit():
            char = msvcrt.getwch()
            if char == "\r":  # Enter pressed
                command = buffer.strip().lower()
                buffer = ""  # reset buffer

                if command == "click":
                    for stream in streams:
                        filename = stream.save_frame()
                        if filename:
                            print(f"📸 Saved: {filename}")
                            img = cv2.imread(filename)
                            processed, detected_colors = detect_colors(img.copy())
                            cv2.imshow(f"Detection Tank {stream.index+1}", cv2.resize(processed, (600, 400)))
                            if detected_colors:
                                print(f"Colors from Tank {stream.index+1}:")
                                for cname, hex_code in detected_colors:
                                    print(f"   - {cname} ({hex_code})")

                elif command == "quit":
                    for stream in streams:
                        stream.stop()
                    cv2.destroyAllWindows()
                    return
            else:
                buffer += char

        if cv2.waitKey(1) & 0xFF == 27:  # Esc key exits too
            break

    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()

# ---------- Run ----------
if __name__ == "__main__":
    run_multicamera(max_cams=len(SAVE_DIRS))
