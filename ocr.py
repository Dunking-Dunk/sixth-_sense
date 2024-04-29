import cv2
import easyocr


class OCR:
    def __init__(self, voice_assistant,languages=['en', 'hi'], use_gpu=False):
        self.voice_assistant = voice_assistant
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def extract_text_from_image(self, image_path):
        image = cv2.imread(image_path)
        new_width = 800
        new_height = 600
        resized_image = cv2.resize(image, (new_width, new_height))
        result = self.reader.readtext(resized_image)
        text_concatenated = ''.join([item[1] for item in result])
        if len(text_concatenated) > 0:
            self.voice_assistant.speak(text_concatenated)
        else:
            self.voice_assistant.speak("Sorry, I can't answer that question.")


# ocr = OCR()
# ocr.extract_text_from_image("/Users/danny/Downloads/heic-5/_3.jpg")