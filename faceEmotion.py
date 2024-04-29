import cv2
import numpy as np
import os
import tensorflow as tf
from skimage.transform import resize
import face_recognition
from pycoral.utils.edgetpu import make_interpreter

class FaceEmotion():
    def __init__(self,voice_assistant):
        self.voice_assistant = voice_assistant
        # self.cap = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.dummy_names = []
        self.known_face_names = []
        self.found_name = None
        self.name = None
        self.emo = None
        self.interpreter = make_interpreter("model/models/modelemotion_edgetpu.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        directory = 'faces/'
        
        for filename in os.listdir(directory):
            if not filename.startswith('.'):
                name = filename[:-5]
                self.known_face_names.append(name)
                image = face_recognition.load_image_file(f"{directory}{filename}")
                encodings = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(encodings)

    def preprocess_img(self, raw):
        img = resize(raw, (200, 200, 3))
        img = np.expand_dims(img, axis=0)
        if np.max(img) > 1:
            img = img / 255.0
        return img

    def brain(self, raw, x, y, w, h):
        img = self.crop_center(raw, x, y, w, h)
        img = self.preprocess_img(img)
        self.interpreter.set_tensor(self.input_details['index'], img.astype(np.float32))
        self.interpreter.invoke()
        res = self.interpreter.get_tensor(self.output_details['index'])
        classes = np.argmax(res, axis=1)
        emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprised']
        return emotions[classes[0]]

    def crop_center(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    def detect_faces(self, frame):
        cascPath = "model/models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150)
        )


        for (x, y, w, h) in faces:
            self.emo = self.brain(gray, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, self.emo, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,cv2.LINE_AA)
            cv2.putText(frame, self.name, (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,cv2.LINE_4)
           
            small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                self.name = "Unknown"

                if any(matches):
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        self.name = self.known_face_names[best_match_index]
                        if len(self.dummy_names) > 1:
                            self.dummy_names = []
                        if self.name in self.dummy_names:
                            continue
                        else:
                            self.dummy_names.append(self.name)
                            print(f"{self.name} is in front of you and he is {self.emo}")
                            self.voice_assistant.speak(f"{self.name} is in front of you and he is {self.emo}")
                    return frame
                    
    def preprocess_frame(self, frame):
        resized_frame = cv2.resize(frame, (64, 64))
        if resized_frame.shape[-1] == 3:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = resized_frame.astype('float32') / 255.0
        input_data = np.expand_dims(normalized_frame, axis=0)
        return input_data

# facemo = FaceEmotion()
# facemo.detect_faces()
