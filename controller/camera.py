import cv2
import numpy as np
import logging

class Camera:
    def __init__(self, camera_id: int, preview: bool):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            logging.error(f"Cannot open camera with ID {camera_id}")
            raise IOError(f"Cannot open camera with ID {camera_id}")
        self.preview = preview

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame

    def find_reference_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) < 1000:
            return None

        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if self.preview:
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        return box

    def show_preview(self, frame):
        if self.preview:
            cv2.imshow('Camera Preview', frame)
            if cv2.waitKey(1) == ord('q'):
                return False
        return True

    def release(self):
        self.cap.release()
        if self.preview:
            cv2.destroyAllWindows()
