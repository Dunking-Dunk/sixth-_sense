import cv2

class FaceDistance:
    def __init__(self, known_distance, known_width,  cascade_path = "model/models/haarcascade_frontalface_default.xml",ref_image_path = "ref_image.jpeg"):
        self.known_distance = known_distance
        self.known_width = known_width
        self.ref_image_path = ref_image_path
        self.cascade_path = cascade_path
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        self.focal_length = self.calculate_focal_length()
        self.found_distance = None

    def calculate_focal_length(self):
        ref_image = cv2.imread(self.ref_image_path)
        ref_image_face_width = self.face_data(ref_image)
        return self.focal_length_finder(self.known_distance, self.known_width, ref_image_face_width)

    def focal_length_finder(self, measured_distance, real_width, width_in_rf_image):
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    def distance_finder(self, face_width_in_frame):
        distance = (self.known_width * self.focal_length) / face_width_in_frame
        return distance

    def face_data(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_image, 1.3, 5)
        face_width = 0

        for (x, y, h, w) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_width = w

        return face_width

    def detect_distance(self,frame):
        # cap = cv2.VideoCapture(0)
        # cap.set(3, 1280)
        # cap.set(4, 720)

        
        # _, frame = cap.read()
        face_width_in_frame = self.face_data(frame)

        if face_width_in_frame != 0:
            distance = self.distance_finder(face_width_in_frame)

            self.found_distance = distance
            return self.found_distance
        else: 
            return 0