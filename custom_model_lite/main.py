from gesture.gesrec import HandGestureRecognition,get_args

handrecognition = HandGestureRecognition

def gestureRecognition():
    args = get_args()
    hand_gesture_recognition = HandGestureRecognition(args)
    hand_gesture_recognition.run()

if __name__ == "__main__":
    gestureRecognition()