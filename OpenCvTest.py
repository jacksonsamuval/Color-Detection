import cv2
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import threading

# ---------- Settings ----------
CSV_PATH = "colornames.csv"
MIN_CONTOUR_AREA = 1000
SAT_THRESH = 80
VAL_THRESH = 80
BACKEND_URL = "http://10.160.114.20:8080/opencv/save"

SAVE_DIRS = ["captures/tank1", "captures/tank2", "captures/tank3"]
for d in SAVE_DIRS:
    os.makedirs(d, exist_ok=True)

# ---------- Load colors CSV ----------
def hex_to_rgb(hex_str):
    hex_str = str(hex_str).lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

colors_df = pd.read_csv(CSV_PATH)
colors_df[["R", "G", "B"]] = colors_df["hex"].apply(lambda x: pd.Series(hex_to_rgb(x)))

def get_color_name(b, g, r):
    min_dist = 1e9
    cname, hex_code = "Unknown", "#000000"
    for i in range(len(colors_df)):
        dr, dg, db = int(colors_df.loc[i, "R"]), int(colors_df.loc[i, "G"]), int(colors_df.loc[i, "B"])
        dist = abs(r - dr) + abs(g - dg) + abs(b - db)
        if dist < min_dist:
            min_dist = dist
            cname = colors_df.loc[i, "name"]
            hex_code = colors_df.loc[i, "hex"]
    return cname, hex_code

# ---------- Send to backend ----------
def send_to_backend(tank_name, color_name, hex_code, file_path):
    date_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
        data = {
            "tankName": tank_name,
            "colorName": color_name,
            "hex": hex_code,
            "dateTime": date_time
        }
        try:
            response = requests.post(BACKEND_URL, data=data, files=files)
            print(f"Backend response ({tank_name}):", response.status_code, response.text)
        except Exception as e:
            print(f"Error sending to backend for {tank_name}:", e)

# ---------- Color detection ----------
def detect_colors(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
        return []

    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:,:,1].astype(np.int32), hsv[:,:,2].astype(np.int32)
    mask = (sat >= SAT_THRESH) & (val >= VAL_THRESH)

    if np.any(mask):
        idx_flat = np.argmax((sat * mask)*1000 + val*mask)
        idx = np.unravel_index(idx_flat, sat.shape)
        py, px = idx
        b, g, r = roi[py, px].tolist()
    else:
        cy, cx = h//2, w//2
        b, g, r = roi[cy, cx].tolist()

    color_name, hex_code = get_color_name(b, g, r)
    return [(color_name, hex_code)]

# ---------- Main ----------
def capture_photos():
    tank_names = ["TNK123", "TNK124", "TNK125"]
    current_tank_index = 0  # Start with tank 1
    
    # Initialize only one camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera 0 not available")
        return
    
    print("✅ Camera 0 ready")
    print(f"Current tank: {tank_names[current_tank_index]}")

    # Shared variable to trigger capture
    capture_flag = threading.Event()
    quit_flag = False

    def input_thread():
        nonlocal quit_flag
        while True:
            command = input("Type 'click' to capture current tank or 'quit' to exit: ").strip().lower()
            if command == "click":
                capture_flag.set()
            elif command == "quit":
                quit_flag = True
                break

    input_thread = threading.Thread(target=input_thread, daemon=True)
    input_thread.start()

    print("Waiting for 'click' command...")
    
    while not quit_flag:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame from camera")
            threading.Event().wait(0.1)
            continue

        # If capture requested
        if capture_flag.is_set():
            current_tank_name = tank_names[current_tank_index]
            print(f"\n--- Processing {current_tank_name} ---")
            
            # Capture and save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIRS[current_tank_index], f"{current_tank_name}_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"✅ Image captured for {current_tank_name}")
            
            # Detect colors
            detected_colors = detect_colors(frame)
            if detected_colors:
                cname, hex_code = detected_colors[0]
                print(f"Detected color: {cname} ({hex_code}) for {current_tank_name}")
                
                # Send to backend
                print(f"Sending to backend for {current_tank_name}...")
                send_to_backend(current_tank_name, cname, hex_code, save_path)
                print(f"✅ Data sent for {current_tank_name}")
            else:
                print(f"No colors detected for {current_tank_name}")
            
            print(f"--- Completed {current_tank_name} ---\n")
            
            # Move to next tank (cycle through 0, 1, 2)
            current_tank_index = (current_tank_index + 1) % 3
            print(f"Next tank: {tank_names[current_tank_index]}")
            
            capture_flag.clear()
            print("Waiting for next 'click' command...")

        # Small delay to prevent high CPU usage
        threading.Event().wait(0.1)

    # Clean up
    cap.release()
    print("Camera released. Program exited.")

if __name__ == "__main__":
    capture_photos()