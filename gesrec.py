import argparse
import csv
import copy
import itertools
from collections import Counter, deque
from time import sleep
import cv2 as cv
import mediapipe as mp
import numpy as np
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


class HandGestureRecognition:
    def __init__(self, args):
        self.args = args
        self.cap_width = 960
        self.cap_height = 540
        self.use_static_image_mode = 'store_true'
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.5
        self.use_brect = True
        # self.cap = cv.VideoCapture(0)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        self.load_labels()
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.mode = 0
        self.close_counter = 0

    def load_labels(self):
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    def run(self,image):
        fps = self.cvFpsCalc.get()
        key = 27
        number, self.mode = self.select_mode(key, self.mode)
        # ret, image = self.cap.read()
    
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        val = self.process_results(debug_image, results, number)
        
        debug_image = self.draw_point_history(debug_image, self.point_history)
        debug_image = self.draw_info(debug_image, fps, self.mode, number)
        #cv.imshow('Hand Gesture Recognition', debug_image)
        return val, debug_image
        

    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:
            number = key - 48
        if key == 110:
            mode = 0
        if key == 107:
            mode = 1
        if key == 104:
            mode = 2
        return number, mode

    def process_results(self, debug_image, results, number):
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(debug_image, self.point_history)
                self.logging_csv(number, self.mode, pre_processed_landmark_list, pre_processed_point_history_list)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                
                if hand_sign_id == 1:
                    self.close_counter += 1
                if hand_sign_id == "Not applicable":
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()
                debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                return hand_sign_id
        else:
            self.point_history.append([0, 0])

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list
    
    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

       
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

       
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

       
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)


            # Drawing circles on landmarks
        for index, landmark in enumerate(landmark_point):
            if index == 0:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image
                

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                       1, cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)
        return image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=int, default=0.5)
    args = parser.parse_args()
    return args

# def main():
#     args = get_args()
#     hand_gesture_recognition = HandGestureRecognition(device=0, width=960, height=540,min_detection_confidence=0.7,min_trracking_confidence=0.5)
#     hand_gesture_recognition.run()

