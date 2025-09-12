# import cv2
# import pandas as pd
# import numpy as np

# # Load colors dataset from CSV
# csv_path = "colornames.csv"
# colors_df = pd.read_csv(csv_path)

# # Convert hex → R, G, B
# def hex_to_rgb(hex_str):
#     hex_str = hex_str.lstrip('#')
#     return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# colors_df[["R", "G", "B"]] = colors_df["hex"].apply(lambda x: pd.Series(hex_to_rgb(x)))

# # Function to get nearest color name from dataset
# def get_color_name(b, g, r):
#     min_dist = 1e9
#     cname = "Unknown"
#     for i in range(len(colors_df)):
#         dr = int(colors_df.loc[i, "R"])
#         dg = int(colors_df.loc[i, "G"])
#         db = int(colors_df.loc[i, "B"])
#         dist = abs(r - dr) + abs(g - dg) + abs(b - db)  # Manhattan distance
#         if dist < min_dist:
#             min_dist = dist
#             cname = colors_df.loc[i, "name"]  # use "name" column from your CSV
#     return cname

# # Open camera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale for contour detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 1000:  # filter small objects
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#             # Get average BGR inside bounding box
#             roi = frame[y:y + h, x:x + w]
#             b, g, r = np.mean(roi[:, :, 0]), np.mean(roi[:, :, 1]), np.mean(roi[:, :, 2])

#             # Match color with CSV dataset
#             color_name = get_color_name(int(b), int(g), int(r))

#             cv2.putText(frame, f"{color_name}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     cv2.imshow("Object & Color Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import pandas as pd
# import numpy as np
# import sys
# import time
# import argparse

# # ---------- Settings / CLI ----------
# parser = argparse.ArgumentParser(description="IP camera color detection (saturated-point)")
# parser.add_argument("--url", type=str, default="", help="Camera URL (rtsp:// or http://). If empty, uses local camera index 0")
# parser.add_argument("--csv", type=str, default="colornames.csv", help="Path to color names CSV (with hex and name columns)")
# parser.add_argument("--min_area", type=int, default=1000, help="Minimum contour area to consider")
# parser.add_argument("--sat_thresh", type=int, default=80, help="Min saturation to accept pixel")
# parser.add_argument("--val_thresh", type=int, default=80, help="Min brightness (value) to accept pixel")
# args = parser.parse_args()

# CAMERA_URL = args.url  # e.g. rtsp://user:pass@192.168.1.50:554/stream1
# CSV_PATH = args.csv
# MIN_CONTOUR_AREA = args.min_area
# SAT_THRESH = args.sat_thresh
# VAL_THRESH = args.val_thresh

# # ---------- Load colors CSV (hex -> R,G,B) ----------
# def hex_to_rgb(hex_str):
#     hex_str = str(hex_str).lstrip('#')
#     if len(hex_str) != 6:
#         raise ValueError(f"Invalid hex: {hex_str}")
#     return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# try:
#     colors_df = pd.read_csv(CSV_PATH)
# except Exception as e:
#     print("Failed to load CSV:", e)
#     sys.exit(1)

# if "hex" not in colors_df.columns or "name" not in colors_df.columns:
#     print("CSV must contain 'hex' and 'name' columns.")
#     sys.exit(1)

# colors_df[["R", "G", "B"]] = colors_df["hex"].apply(lambda x: pd.Series(hex_to_rgb(x)))

# # simple matcher (Manhattan). You can replace with Lab Euclidean later if desired.
# def get_color_name(b, g, r):
#     min_dist = 1e9
#     cname = "Unknown"
#     hex_code = "#000000"
#     for i in range(len(colors_df)):
#         dr = int(colors_df.loc[i, "R"])
#         dg = int(colors_df.loc[i, "G"])
#         db = int(colors_df.loc[i, "B"])
#         dist = abs(r - dr) + abs(g - dg) + abs(b - db)
#         if dist < min_dist:
#             min_dist = dist
#             cname = colors_df.loc[i, "name"]
#             hex_code = colors_df.loc[i, "hex"]
#     return cname, hex_code

# # ---------- Open camera ----------
# def open_capture(url):
#     if url:
#         print("Trying to open camera URL:", url)
#         cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # allow FFMPEG backend when available
#         # fall back without backend flag if needed:
#         if not cap.isOpened():
#             cap = cv2.VideoCapture(url)
#     else:
#         cap = cv2.VideoCapture(0)
#     return cap

# cap = open_capture(CAMERA_URL)
# if not cap.isOpened():
#     print("Warning: cannot open camera. If using an IP camera, verify URL, credentials and network.")
#     sys.exit(1)

# # set timeout/read attempt behaviour
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# print("Press 'q' to quit. 's' to save snapshot.jpg")

# # ---------- Main loop ----------
# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         print("Frame not received — retrying in 0.5s...")
#         time.sleep(0.5)
#         # try to reconnect once
#         cap.release()
#         cap = open_capture(CAMERA_URL)
#         if not cap.isOpened():
#             print("Reconnect failed; will keep retrying...")
#             time.sleep(1)
#             continue
#         else:
#             continue

#     # pre-process & simple contour detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (7, 7), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # morphological clean-up
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < MIN_CONTOUR_AREA:
#             continue
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi = frame[y:y + h, x:x + w]
#         if roi.size == 0:
#             continue

#         # convert ROI to HSV and find most saturated+bright pixel
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         sat = hsv[:, :, 1].astype(np.int32)
#         val = hsv[:, :, 2].astype(np.int32)
#         mask = (sat >= SAT_THRESH) & (val >= VAL_THRESH)

#         if np.any(mask):
#             # pick pixel with maximum saturation (tie-breaker: higher value)
#             idx_flat = np.argmax((sat * mask) * 1000 + val * mask)
#             idx = np.unravel_index(idx_flat, sat.shape)
#             py, px = int(idx[0]), int(idx[1])
#             b, g, r = roi[py, px].tolist()
#             # draw dot on ROI to show sampling point
#             cv2.circle(frame, (x + px, y + py), 4, (255, 255, 255), -1)
#         else:
#             # fallback: sample center pixel
#             cy, cx = h // 2, w // 2
#             b, g, r = roi[cy, cx].tolist()

#         color_name, hex_code = get_color_name(int(b), int(g), int(r))
#         label = f"{color_name} {hex_code}"
#         cv2.rectangle(frame, (x, y - 26), (x + 220, y), (0, 0, 0), -1)  # background for text
#         cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     cv2.imshow("IP Color Detector", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     if key == ord('s'):
#         cv2.imwrite("snapshot.jpg", frame)
#         print("Saved snapshot.jpg")

# cap.release()
# cv2.destroyAllWindows()