import cv2
import pandas as pd
import numpy as np
import sys
import time
import threading
import queue

# ---------- Settings ----------
CSV_PATH = "colornames.csv"
MIN_CONTOUR_AREA = 1000
SAT_THRESH = 80
VAL_THRESH = 80

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
    def __init__(self, max_test=5):
        self.cap = None
        self.frame = None
        self.stopped = False

        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"✅ Using camera index {i}")
                self.cap = cap
                break

        if self.cap is None:
            print("❌ No working camera found!")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# ---------- Processing ----------
def process_frame(frame, save_path="snapshot.jpg"):
    frame = cv2.resize(frame, (640, 480))
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
        cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(save_path, frame)
    print(f"\n📸 Saved {save_path}")
    if detected_colors:
        print("🎨 Detected colors:")
        for cname, hex_code in detected_colors:
            print(f"   - {cname} ({hex_code})")
    else:
        print("⚠️ No significant colors detected!")

# ---------- Input Thread ----------
def input_listener(cmd_queue):
    while True:
        command = input().strip().lower()
        cmd_queue.put(command)
        if command == "quit":
            break

# ---------- Main ----------
stream = CameraStream()
time.sleep(1)

cmd_queue = queue.Queue()
threading.Thread(target=input_listener, args=(cmd_queue,), daemon=True).start()

print("Type 'click' to capture, 'quit' to exit.\n")

while True:
    frame = stream.read()
    if frame is None:
        time.sleep(0.01)
        continue

    cv2.imshow("Live Preview", cv2.resize(frame, (640, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if a command was entered
    if not cmd_queue.empty():
        command = cmd_queue.get()
        if command == "click":
            process_frame(frame.copy(), "snapshot.jpg")
        elif command == "quit":
            break

stream.stop()
cv2.destroyAllWindows()