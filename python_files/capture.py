import os
import cv2
import time

class WebcamCapture:
    def __init__(self,voice_assistant, camera_index=0):
        self.voice_assistant = voice_assistant
        self.camera_index = camera_index
        self.camera = None

    def start_capture(self):
        self.camera = cv2.VideoCapture(self.camera_index)

    def capture_frame(self, save_path=None):
        success, frame = self.camera.read()
        if success:
            # cv2.imshow("Webcam", frame)
            if save_path:
                cv2.imwrite(save_path, frame)
            if cv2.waitKey(1):
                return False
        return True

    def stop_capture(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def capture_photo(self, save_path="images/captured_image.jpg"):
        self.start_capture()
        time.sleep(2)
        while True:
            if not self.capture_frame(save_path):
                self.voice_assistant.speak("Photo Captured")
                break
        self.stop_capture()
        return os.path.abspath(save_path)
# webcam = WebcamCapture()
# image_path = webcam.capture_photo()
