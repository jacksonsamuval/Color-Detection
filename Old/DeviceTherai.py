# import cv2
# import pandas as pd
# import numpy as np
# import os
# import requests
# from datetime import datetime

# # ---------- Settings ----------
# CSV_PATH = "colornames.csv"
# MIN_CONTOUR_AREA = 1000
# SAT_THRESH = 80
# VAL_THRESH = 80
# BACKEND_URL = "http://10.190.225.20:8080/opencv/save"

# SAVE_DIRS = ["captures/tank1", "captures/tank2"]
# for d in SAVE_DIRS:
#     os.makedirs(d, exist_ok=True)

# # ---------- Load colors CSV ----------
# def hex_to_rgb(hex_str):
#     hex_str = str(hex_str).lstrip('#')
#     return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# colors_df = pd.read_csv(CSV_PATH)
# colors_df[["R", "G", "B"]] = colors_df["hex"].apply(lambda x: pd.Series(hex_to_rgb(x)))

# def get_color_name(b, g, r):
#     min_dist = 1e9
#     cname, hex_code = "Unknown", "#000000"
#     for i in range(len(colors_df)):
#         dr, dg, db = int(colors_df.loc[i, "R"]), int(colors_df.loc[i, "G"]), int(colors_df.loc[i, "B"])
#         dist = abs(r - dr) + abs(g - dg) + abs(b - db)
#         if dist < min_dist:
#             min_dist = dist
#             cname = colors_df.loc[i, "name"]
#             hex_code = colors_df.loc[i, "hex"]
#     return cname, hex_code

# # ---------- Send to backend ----------
# def send_to_backend(tank_name, color_name, hex_code, file_path):
#     date_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
#     with open(file_path, "rb") as f:
#         files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
#         data = {
#             "tankName": tank_name,
#             "colorName": color_name,
#             "hex": hex_code,
#             "dateTime": date_time
#         }
#         try:
#             response = requests.post(BACKEND_URL, data=data, files=files)
#             print(f"Backend response ({tank_name}):", response.status_code, response.text)
#         except Exception as e:
#             print(f"Error sending to backend for {tank_name}:", e)

# # ---------- Color detection ----------
# def detect_colors(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3,3), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     detected_colors = []

#     for cnt in contours:
#         if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
#             continue
#         x, y, w, h = cv2.boundingRect(cnt)
#         roi = frame[y:y+h, x:x+w]
#         if roi.size == 0:
#             continue
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         sat, val = hsv[:,:,1].astype(np.int32), hsv[:,:,2].astype(np.int32)
#         mask = (sat >= SAT_THRESH) & (val >= VAL_THRESH)
#         if np.any(mask):
#             idx_flat = np.argmax((sat * mask)*1000 + val*mask)
#             idx = np.unravel_index(idx_flat, sat.shape)
#             py, px = idx
#             b, g, r = roi[py, px].tolist()
#         else:
#             cy, cx = h//2, w//2
#             b, g, r = roi[cy, cx].tolist()
#         color_name, hex_code = get_color_name(b, g, r)
#         detected_colors.append((color_name, hex_code))
#     return detected_colors

# # ---------- Main ----------
# def capture_photos():
#     # Only two cameras with fixed tank names
#     tank_names = ["TNK123", "TNK124"]

#     cameras = []
#     for i in range(2):
#         cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#         if cap.isOpened():
#             print(f"✅ Camera {i} ready ({tank_names[i]})")
#             cameras.append((i, cap))
#         else:
#             print(f"❌ Camera {i} not available")

#     if not cameras:
#         print("No cameras found.")
#         return

#     while True:
#         command = input("Type 'click' to capture photos or 'quit' to exit: ").strip().lower()
#         if command == "quit":
#             break
#         elif command == "click":
#             # Sequential capture and send per camera
#             for cam_index, cap in cameras:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print(f"Failed to capture camera {cam_index}")
#                     continue

#                 # Save image
#                 save_path = os.path.join(SAVE_DIRS[cam_index], f"capture_{datetime.now().strftime('%H%M%S')}.jpg")
#                 cv2.imwrite(save_path, frame)
#                 print(f"📸 Saved: {save_path}")

#                 # Detect colors
#                 detected_colors = detect_colors(frame)
#                 if not detected_colors:
#                     print(f"No colors detected for {tank_names[cam_index]}")
#                     continue

#                 # Send detected colors one by one and wait for response
#                 for cname, hex_code in detected_colors:
#                     tank_name = tank_names[cam_index]
#                     print(f"Sending {cname} ({hex_code}) for {tank_name}...")
#                     send_to_backend(tank_name, cname, hex_code, save_path)
#                     print(f"✅ Response received for {tank_name}")

#             print("✅ All cameras processed for this click.\n")

#     # Release cameras
#     for _, cap in cameras:
#         cap.release()

# if __name__ == "__main__":
#     capture_photos()



import cv2
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime

# ---------- Settings ----------
CSV_PATH = "colornames.csv"
MIN_CONTOUR_AREA = 1000
SAT_THRESH = 80
VAL_THRESH = 80
BACKEND_URL = "http://10.190.225.20:8080/opencv/save"

SAVE_DIRS = ["captures/tank1", "captures/tank2"]
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
    """
    Detect only one dominant color per frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Use the largest contour to determine dominant color
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
    return [(color_name, hex_code)]  # Only one dominant color

# ---------- Main ----------
def capture_photos():
    tank_names = ["TNK123", "TNK124"]  # Fixed tank names

    cameras = []
    for i in range(2):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera {i} ready ({tank_names[i]})")
            cameras.append((i, cap))
        else:
            print(f"❌ Camera {i} not available")

    if not cameras:
        print("No cameras found.")
        return

    while True:
        command = input("Type 'click' to capture photos or 'quit' to exit: ").strip().lower()
        if command == "quit":
            break
        elif command == "click":
            # Capture and process each camera sequentially
            for cam_index, cap in cameras:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture camera {cam_index}")
                    continue

                save_path = os.path.join(SAVE_DIRS[cam_index], f"capture_{datetime.now().strftime('%H%M%S')}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📸 Saved: {save_path}")

                detected_colors = detect_colors(frame)
                if not detected_colors:
                    print(f"No colors detected for {tank_names[cam_index]}")
                    continue

                # Send the single detected color
                cname, hex_code = detected_colors[0]
                print(f"Sending {cname} ({hex_code}) for {tank_names[cam_index]}...")
                send_to_backend(tank_names[cam_index], cname, hex_code, save_path)
                print(f"✅ Response received for {tank_names[cam_index]}")

            print("✅ All cameras processed for this click.\n")

    # Release cameras
    for _, cap in cameras:
        cap.release()

if __name__ == "__main__":
    capture_photos()