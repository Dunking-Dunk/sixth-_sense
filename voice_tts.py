import pyttsx3

class VoiceAssistant():
    def __init__(self) :
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', self.tts_engine.getProperty('rate') - 15)
        self.tts_engine.setProperty('volume', self.tts_engine.getProperty('volume') + 1)
    def speak(self, text):
        """Speak the given text using the TTS engine"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

