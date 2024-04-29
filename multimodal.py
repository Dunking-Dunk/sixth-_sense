from faceEmotion import FaceEmotion
from gesrec import HandGestureRecognition
from detection import VideoStream,Detector,argparse
from time import sleep
from voice_tts import VoiceAssistant
import cv2
from picamera2 import Picamera2
import time
import threading
import numpy as np
from main import midas

voice_assistant = VoiceAssistant()

class VideoStream():
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)},controls={"FrameRate": 8,"FrameDurationLimits": (40000, 40000)}))
        self.picam2.start()
        self.getFrame()
    
    def getFrame(self):
        self.frame = self.picam2.capture_array()
        return self.frame


class multimodal_perception():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true', default=True)
        args = parser.parse_args()
        self.hand_gesture_recognition =HandGestureRecognition(args)
        self.faceEmotion = FaceEmotion(voice_assistant)
        self.detector = Detector(voice_assistant,args.modeldir, args.graph, args.labels, args.threshold, args.resolution, args.edgetpu)
        
    def run(self, frame1):
        frame = np.array(frame1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.detector.width, self.detector.height))
        #t1 = time.perf_counter(), time.process_time()
    
        hand, frame = self.hand_gesture_recognition.run(frame)
        self.faceEmotion.detect_faces(frame)
        
        if (hand == 1):
            boxes, classes, scores = self.detector.detect_objects(frame_resized)
           
            frame = self.detector.draw_boxes(frame, boxes, classes, scores)

        #t2 = time.perf_counter(), time.process_time()

        #print(f" Real time: {t2[0] - t1[0]:.2f} seconds")    
        #print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")

        #cv2.imshow('Multimodal Perception', frame)




#multi = multimodal_perception()
video = VideoStream()

#def process1():
    #while True:
        #video.getFrame()

def process2():
   multi = multimodal_perception()
   while True:
       video.getFrame()
       multi.run(video.frame)
       
def process3():
   mid = midas()
   while True:
       mid.run(video.frame)

if __name__ == "__main__":
    #t1 = threading.Thread(target=process1)
    t2 = threading.Thread(target=process2)
    t3 = threading.Thread(target=process3)
    #t1.start()
    t2.start()
    t3.start()
    #t1.join()
    t2.join()
    t3.join()
   
