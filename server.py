import firebase_admin
from firebase_admin import credentials,  db
from firebase_admin import firestore
import datetime
import re
import requests
import os
from time import time


class FirebaseManager:
    def __init__(self, voice_assistant,credential_file = 'secret.json'):
        self.voice_assistant = voice_assistant
        self.secret = 'Wert'
        self.user_id = 'Wert'
        cred = credentials.Certificate(credential_file)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        #getMessage and getPeoples get initialized when the class is created
        self.getMessage()
        self.getPeoples()
    
    # send message message command
    def sendMessage(self, Message):
        collection = self.db.collection('visionUser').document(self.secret).collection('messages')
        current_datetime = datetime.datetime.now()
        timestamp = current_datetime.timestamp()
        collection.add({
            "message": Message,
            "timestamp": time() * 1000
        })
        self.voice_assistant.speak('Message sent')

    #just add the text to speech command here both getMessage and getPeoples get initialized when the class is created
    def getMessage(self):
        def lol(data,d,h): 
            caregiver_message = data[-1]._data

            current_time = datetime.datetime.now().timestamp()
            current_time = current_time * 1000
            if (caregiver_message['caregiver'] and caregiver_message['timestamp'] + 5000 > current_time):
                #add voice message here
                self.voice_assistant.speak('message from {}: {}'.format(caregiver_message['name'],caregiver_message['message'] ))
        
        self.db.collection('visionUser').document(self.secret).collection('messages').order_by('timestamp').on_snapshot(lol)
      
    # use this command in piApi getLocation to send gps coordinate to server
    def sendGPS(self, user_coord):
        collection = self.db.collection('visionUser').document(self.secret)
        collection.update({
            "coords": firestore.GeoPoint(user_coord[1], user_coord[0])
        })
    
    def compare_strings(self,string1, string2):
        pattern = re.compile(string2)
        match = re.search(pattern, string1)
        return match

    #requires name
    def checkContacts(self, name):
        res = self.db.collection('visionUser').document(self.secret).collection('contacts').get()
        
        for i in res:
            contact = i._data
            if self.compare_strings(contact['name'].lower(), name.lower()):
                #add voice message here
                self.voice_assistant.speak('{} Contact found'.format(name))
                return contact
        self.voice_assistant.speak('No contact found')

    def downloadImage(self, user):
        img_data = requests.get(user['image']).content
        with open('people/{}.jpg'.format(user['name']), 'wb') as handler:
            handler.write(img_data)

    #For face recognition 
    def getPeoples(self):
        def getData(data,d,h):
            for i in data:
                if (os.path.isfile('people/{}.jpg'.format(i._data['name'])) == False):
                    self.downloadImage(i._data)
                
        self.db.collection('visionUser').document(self.secret).collection('peoples').on_snapshot(getData)


   
# dba = FirebaseManager()
