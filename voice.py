import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime
from queue import Queue
from time import sleep
from sys import platform
import socket
import pyttsx3
import sounddevice

class VoiceAssistant:
    def __init__(self):
        self.said = ''
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 100
        self.recorder.dynamic_energy_threshold = False
        self.source = None;
        self.data_queue = Queue()
        self.setup_mic()
            
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', self.tts_engine.getProperty('rate') - 15)
        self.tts_engine.setProperty('volume', self.tts_engine.getProperty('volume') + 1)
        
            
    def selectModel(self):
         if self.is_system_offline():
            return self.whisper_get_audio
         else:
            return self.google_get_audiospeech
        
    def is_system_offline(self):
        try:
            # Attempt to create a socket connection to a known external server (Google DNS).
            socket.create_connection(("8.8.8.8", 53))
            return False
        except OSError:
            return True
            
    def setup_mic(self):
         if 'linux' in platform:
            mic_name = 'pulse'
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
         else:
            self.source = sr.Microphone(sample_rate=16000)
        

    def google_get_audiospeech(self):
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        def record_callback(_, audio:sr.AudioData) -> None:
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio
            try:
                said = self.recorder.recognize_google(data)
                self.said = said
                print(self.said)
            except sr.UnknownValueError:
                print("Could not understand audio")
                #return "Could not understand audio"
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                #return "Could not request results; {0}".format(e)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        #self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=0)
        with self.source as source: 
            audio = self.recorder.listen(source)
            try:
                said = self.recorder.recognize_google(audio)
                return said
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
        return ''
        
    def getText():
        text = self.said
        self.said = ''
        return text

    def whisper_get_audio(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="tiny", help="Model to use",
                            choices=["tiny", "base", "small", "medium", "large"])
        parser.add_argument("--non_english", action='store_true',
                            help="Don't use the english model.")
        parser.add_argument("--energy_threshold", default=10000,
                            help="Energy level for mic to detect.", type=int)
        parser.add_argument("--record_timeout", default=0,
                            help="How real time the recording is in seconds.", type=float)
        parser.add_argument("--phrase_timeout", default=0,
                            help="How much empty space between recordings before we "
                                "consider it a new line in the transcription.", type=float)
       
        args = parser.parse_args()


        model = args.model
        if args.model != "small" and not args.non_english:
            model = model + ".en"
            print(model)
        audio_model = whisper.load_model(model)

        record_timeout = args.record_timeout
        phrase_timeout = args.phrase_timeout

        transcription = ['']

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        def record_callback(_, audio:sr.AudioData) -> None:
        
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            self.data_queue.put(data)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        #self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=0)

        # Cue the user that we're ready to go.
        print("Model loaded.\n")
        with self.source as source: 
            audio_data = self.recorder.listen(source).get_raw_data()
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()
            return text
        return ''
 

    def speak(self, text):
        """Speak the given text using the TTS engine"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

if __name__ == "__main__":
     print("Voice Assistant")
     assistant = VoiceAssistant()
     stt = assistant.selectModel()
     while True:
         text = stt()
         for i in range(10):
             print('*' * i)
         print(text)
         for i in range(10):
             print('*' * i)


