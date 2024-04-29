import datetime

class DateAndTime():

    def __init__(self, voice_assistant):
        self.voice_assistant = voice_assistant
        self.time = None
        self.date = None

    def get_time(self):
        now = datetime.datetime.now()
        h = now.strftime("%H")
        h = int(h)
        min = now.strftime("%M")

        if h <= 12:
            ap = "A M"
        else:
            ap = "P M"

        if h == 13:
            h = 1
        elif h == 14:
            h = 2
        elif h == 15:
            h = 3
        elif h == 16:
            h = 4
        elif h == 17:
            h = 5
        elif h == 18:
            h = 6
        elif h == 19:
            h = 7
        elif h == 20:
            h = 8
        elif h == 21:
            h = 8
        elif h == 22:
            h = 10
        elif h == 23:
            h = 11
        elif h == 24:
            h = 12

        self.time = f"The time is {h} {min} {ap}"
        self.voice_assistant.speak(self.time)
        return

    def get_date(self):
        now = datetime.datetime.now()
        m = int(now.strftime("%m"))
        if m == 1:
            month = "January"
        elif m == 2:
            month = "Feburary"
        elif m == 3:
            month = "March"
        elif m == 4:
            month = "April"
        elif m == 5:
            month = "May"
        elif m == 6:
            month = "June"
        elif m == 7:
            month = "July"
        elif m == 8:
            month = "August"
        elif m == 9:
            month = "September"
        elif m == 10:
            month = "October"
        elif m == 11:
            month = "November"
        elif m == 12:
            month = "December"

        year = now.strftime("%d %Y")
        self.date = f"Today is {month} {year}"
        self.voice_assistant.speak(self.date)
        return