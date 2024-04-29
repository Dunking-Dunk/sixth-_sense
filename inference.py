import time
import cv2
import numpy as np
import urllib.request
import os.path
from pycoral.utils.edgetpu import make_interpreter
from tensorflow.lite.python.interpreter import Interpreter

class midasDepthEstimator:
    def __init__(self):
        self.i = 0
        self.fps = 0
        #self.timeLastPrediction = time.time()
        self.frameCounter = 0

        # Initialize model
        self.initializeModel()

    def initializeModel(self):
        modelPath = 'midas/models1/midasModel.tflite' ### 4fps
        # modelPath = "/Users/danny/Desktop/sixth_sense/python files/midas/models/model_float16_quant.tflite" ### 1 fps
        # modelPath = "/Users/danny/Downloads/saved_model_480x640/model_float32.tflite" ### 0.7 fps
        # modelPath = "/Users/danny/Downloads/saved_model_480x640/model_dynamic_range_quant.tflite" #### 1.5 fpsS
        # modelPath = "/Users/danny/Downloads/saved_model_480x640/model_weight_quant.tflite" ### 1.5 fps
        # modelPath = "/Users/danny/Desktop/sixth_sense/python files/midas/models/fastdepth/model_full_integer_quant.tflite" ### 10 fps
        # modelPath = "/Users/danny/Downloads/depthestimatemodel/model_full_integer_quant.tflite" #### 3.5 fps
        # modelPath = "/Users/danny/Downloads/pydnet2_480x640/model_integer_quant.tflite"
        # modelPath = "/Users/danny/Downloads/megadepth/model_integer_quant.tflite" ### 2.2fps
        # modelPath = "/Users/danny/Downloads/tflite_from_saved_model/model_full_integer_quant.tflite"#### 11 fps best so far
        # modelPath = "/Users/danny/Downloads/stereonet_240x320/model_float32.tflite" ### 1 fps
        # modelPath = "/Users/danny/Downloads/nyu.tflite"   
        # modelPath = "/Users/danny/Downloads/resources/03_integer_quantization/dense_depth_nyu_480x640_integer_quant.tflite"
        # if not os.path.isfile(modelPath):
        #     url = 'https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1?lite-format=tflite'
        #     urllib.request.urlretrieve(url, modelPath)

        self.interpreter = Interpreter(modelPath, num_threads=1)
        self.interpreter.allocate_tensors()

        # Get model info
        self.getModelInputDetails()
        self.getModelOutputDetails()

    def estimateDepth(self, image):
        input_tensor = self.prepareInputForInference(image)

        # Perform inference on the image
        rawDisparity = self.inference(input_tensor)

        # Normalize and resize raw disparity
        processedDisparity = self.processRawDisparity(rawDisparity, image.shape)

        # Draw depth image
        colorizedDisparity = self.drawDepth(processedDisparity)

        # Update fps calculator
        #self.updateFps()

        return colorizedDisparity

    def prepareInputForInference(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width, self.img_channels = img.shape

        # Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
        # and 256 x 256 pixels for the back model
        img_input = cv2.resize(img, (self.inputWidth,self.inputHeight),interpolation = cv2.INTER_CUBIC).astype(np.float32)

        # Scale input pixel values to -1 to 1
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        reshape_img = img_input.reshape(1, self.inputHeight, self.inputWidth,3)
        img_input = ((img_input/ 255.0 - mean) / std).astype(np.float32)
        img_input = img_input[np.newaxis,:,:,:]        

        return img_input

    def inference(self, img_input):
        # Peform inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = output.reshape(self.outputHeight, self.outputWidth)

        return output


    def processRawDisparity(self, rawDisparity, img_shape):
        # Normalize estimated depth to have values between 0 and 255
        depth_min = rawDisparity.min()
        depth_max = rawDisparity.max()
        normalizedDisparity = (255 * (rawDisparity - depth_min) / (depth_max - depth_min)).astype("uint8")

        # Resize disparity map to the sam size as the image inference
        estimatedDepth = cv2.resize(normalizedDisparity, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)

        return estimatedDepth

    def drawDepth(self, processedDisparity):
        return cv2.applyColorMap(processedDisparity, cv2.COLORMAP_MAGMA)

    def getModelInputDetails(self):
        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]['shape']
        self.inputHeight = input_shape[1]
        self.inputWidth = input_shape[2]
        self.channels = input_shape[3]

    def getModelOutputDetails(self):
        self.output_details = self.interpreter.get_output_details()
        output_shape = self.output_details[0]['shape']
        self.outputHeight = output_shape[1]
        self.outputWidth = output_shape[2]  

    def updateFps(self):
        updateRate = 1
        self.frameCounter += 1

        # Every updateRate frames calculate the fps based on the ellapsed time
        if self.frameCounter == updateRate:
            timeNow = time.time()
            ellapsedTime = timeNow - self.timeLastPrediction

            self.fps = int(updateRate/ellapsedTime)
            self.frameCounter = 0
            self.timeLastPrediction = timeNow

    def checkDistanceThreshold(self, depth_map, image_width):
    # Convert depth map to meters (assuming depth map values are in a suitable format)
        depth_map_meters = depth_map / 1000.0  # Convert from millimeters to meters

        # Find the minimum depth value (closest object)
        min_depth = np.min(depth_map_meters)
        print(min_depth)

        # Define threshold distance (in meters)
        threshold_distance = 0.1  # Adjust as needed

        # Check if the minimum depth is less than the threshold distance
        if min_depth < threshold_distance:
            # Find the position of the object in the image
            object_position = self.getObjectPosition(depth_map, image_width)
            

            # Trigger warning with directional information
            if object_position == "left":
                self.i = self.i + 1
                # return object_position
                print("Warning object on left",self.i)
            elif object_position == "right":
                self.i = self.i + 1
                print("Warning object on right",self.i)
                # return object_position
            else:
                self.i = self.i + 1
                print("Warning object on center",self.i)
           

    def getObjectPosition(self, depth_map, image_width):
    # Calculate the depth map's midpoint
        # depth_midpoint = depth_map.shape[1] // 2
        depth_midpoint = depth_map.shape[1] // 3

        # Calculate the average depth of the left and right halves of the depth map
        left_depth_avg = np.mean(depth_map[:, :depth_midpoint])
        center_depth_avg = np.mean(depth_map[:, depth_midpoint:])
        right_depth_avg = np.mean(depth_map[:, depth_midpoint:])

        # Determine the position of the object based on the average depths
        if center_depth_avg < right_depth_avg and center_depth_avg > left_depth_avg:
            object_position = "center"
        elif right_depth_avg < left_depth_avg:
            object_position = "left"
        elif left_depth_avg < center_depth_avg:
            object_position = "right"
            

        return object_position
