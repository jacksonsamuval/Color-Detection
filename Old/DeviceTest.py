import cv2
import threading
import time
import sys

# ---------- CameraStream ----------
class CameraStream:
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.frame = None
        self.stopped = False
        self.index = cam_index

        if not self.cap.isOpened():
            print(f"❌ Camera {cam_index} not available.")
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

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

# ---------- Multi-camera handler ----------
def run_multicamera(max_cams=5):
    streams = []
    for i in range(max_cams):
        stream = CameraStream(i)
        if stream.cap:
            streams.append(stream)

    if not streams:
        print("❌ No working cameras found!")
        sys.exit(1)

    print("Press 'q' to quit.\n")

    while True:
        for stream in streams:
            frame = stream.read()
            if frame is not None:
                window_name = f"Camera {stream.index}"
                cv2.imshow(window_name, cv2.resize(frame, (480, 360)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()

# ---------- Main ----------
if __name__ == "__main__":
    run_multicamera(max_cams=5)
