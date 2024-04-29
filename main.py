import cv2
import numpy as np
from inference import midasDepthEstimator
from picamera2 import Picamera2

# Initialize depth estimation model

class midas():
        def __init__(self):
                self.depthEstimator = midasDepthEstimator()
                self.fps = 1 
                self.freq = cv2.getTickFrequency()
                
        def run(self, frame):
            t1 = cv2.getTickCount()
            # Read frame from the webcam
            img = np.array(frame)

            # Estimate depth
            colorDepth = self.depthEstimator.estimateDepth(img)
            
            # Get depth map
            depth_map = self.depthEstimator.inference(self.depthEstimator.prepareInputForInference(img))

            # Check distance threshold
            # depthEstimator.checkDistanceThreshold(depth_map, img.shape[1])  # Pass image width for object position calculation

            # Add the depth image over the color image:
            #combinedImg = cv2.addWeighted(img, 0.7, colorDepth, 0.6, 0)

            # Join the input image, the estimated depth, and the combined image
            #img_out = np.hstack((img, colorDepth, combinedImg))
            # cv2.putText(img_out, "FPS: {}".format(depthEstimator.fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(img_out, "FPS: {}".format(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            # cv2.putText(img)
            # print(fps)
            position = self.depthEstimator.checkDistanceThreshold(depth_map, img.shape[1])
            # if position == "left":
            #     cv2.putText(combinedImg,"WARNING!! object on right", (500, 500),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            # elif position == "right":
            #     cv2.putText(combinedImg,"WARNING!! object on left", (500, 500),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            #cv2.imshow("Depth Image", colorDepth)
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/self.freq
            fps = 1/time1
    



