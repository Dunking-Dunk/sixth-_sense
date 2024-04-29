import requests
from geopy import distance

#coords [lng, lat]
test_user = [80.1923155341089, 13.087384660288127, ]
test_loc = [ 80.1612290030796, 13.081351234586313,]

#first when it is called by caregiver set this user_steo_navigation true
user_step_navigation = True
#if user tells stop navigation
user_step_navigation = False

class NavigateUser:
    def __init__(self,voice_assistant):
        self.accessToken = 'pk.eyJ1IjoiaHVyc3VuIiwiYSI6ImNsc3Q2N2FocjFvbGIyaXBzbWM2Z2w1NGMifQ.tlgUONgceoguAL2oKuqgdQ'
        self.user_step_navigation = False
        self.voice_assistant = voice_assistant
        self.user_coord = [80.18016943867312,13.078225002745956]
    
        # cred = credentials.Certificate("secret.json # Enter you .json file")
        # firebase_admin.initialize_app(cred)
        # db = firestore.client()
        # secret = 'Wret'
        # collection = db.collection('visionUser').document('Wert')  # create collection
            
        
    #user calls for navigation by giving location name eg: 'Navigate Mogappair east grace market'
    def setupNavigation(self, address):
        address_coords_res=requests.get(f"https://api.mapbox.com/search/geocode/v6/forward?q={address}&country=IND&access_token={self.accessToken}")
        address_coords = address_coords_res.json()['features'][0]['geometry']['coordinates']
        print(self.user_coord)
        #now take the address coord and feed it into the direction call
        #provide the coordinates of the user location then address coords
        #coords [lng, lat]
        user_coord = self.user_coord
        res = requests.get(f"https://api.mapbox.com/directions/v5/mapbox/walking/{user_coord[0]}%2C{user_coord[1]}%3B{address_coords[0]}%2C{address_coords[1]}?alternatives=false&continue_straight=true&geometries=geojson&language=en&overview=full&steps=true&access_token={self.accessToken}")

        if (res.status_code == 200):
            self.user_step_navigation = True
            data = res.json()
            self.prev_dist = 0
            routes = data['routes']
            self.total_distance = data['routes'][0]['distance'] 
            self.steps = routes[0]['legs'][0]['steps']
            self.voice_assistant.speak(self.steps[0]['maneuver']['instruction'])
        else:
            self.user_step_navigation = False
            self.voice_assistant.speak('cant find the address call the command again')
        
   
       
    
    def stopNavigation(self):
        self.user_step_navigation = False
        self.voice_assistant.speak('navigation is stopped')

    #call this inside while loop
    def navigate(self, user_coord):
        if (self.user_step_navigation == True and len(self.steps) > 0):
            #call text to text speech model
            loc = self.steps[0]['maneuver']['location']
            dist = distance.distance((loc[1], loc[0]), (user_coord[1], user_coord[0])).meters
            if (dist < 20):
                self.steps.pop(0)
                if (len(self.steps) > 0):
                    self.voice_assistant.speak(self.steps[0]['maneuver']['instruction'] )
                else:
                    self.voice_assistant.speak('You have reached your Destination')
                    self.user_step_navigation = False
            #if (dist > self.prev_dist):
                #self.voice_assistant.speak('Going in wrong Direction')

            self.prev_dist = dist
