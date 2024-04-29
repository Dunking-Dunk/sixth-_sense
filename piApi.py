import time
import serial
import pynmea2
import threading 
from mpu6050 import mpu6050
import math
from gsmmodem.modem import GsmModem, SentSms
from gsmmodem.exceptions import InterruptedException, CommandError
import gpiod

from voice import VoiceAssistant

class piAPI():
	def __init__(self, db,voice_assistant):
		print('pi API initiated')
		self.voice_assistant = voice_assistant
		self.db = db
	
		self.setUpGPS()
		self.user_coord = [0,0]
		self.GPSActive = True
	
		self.setupGyro()
		self.fallDetected = False
			
		self.setupVibrate()
		
		self.GSMActive = False
	
	def send_at(self,command,back,timeout):
		rec_buff = ''
		self.serial.write((command+'\r\n').encode())
		time.sleep(timeout)
		if self.serial.inWaiting():
			time.sleep(0.01 )
			rec_buff = self.serial.read(self.serial.inWaiting())
		if rec_buff != '':
			if back not in rec_buff.decode():
				print(command + ' ERROR')
				print(command + ' back:\t' + rec_buff.decode())
				return 0
			else:
				print(rec_buff.decode())
				return 1
		else:
			print('GPS is not ready')
			return 0
			
	def setUpGPS(self):
		self.serial = serial.Serial('/dev/ttyAMA0',115200, timeout = 3)
		print('Start GPS session...')
		self.send_at('AT+CGPS=1,1','OK',1)
		self.serial.write(('AT+CGPSINFOCFG=1,2\r\n').encode('ISO-8859-1'))
		self.GPSActive = True
	
	def stopGPS(self):
		self.GPSActive = False
		print('Stop GPS session...')
		self.send_at('AT+CGPS=0,1','OK',1)
		self.serial.close()
	
	#to get gps coordinate
	def getLocation(self):
		data = ''
		#self.serial.write(('AT+CGPSINFOCFG=3,2\r\n').encode('ISO-8859-1'))
		data = self.serial.read(self.serial.inWaiting()).decode('ISO-8859-1')
		if data[0:6] == "$GPRMC":
			newmsg = pynmea2.parse(data)
			lat = newmsg.latitude
			lng = newmsg.longitude
			gps = "Latitude=" + str(lat) + "and Longitude=" + str(lng)
			self.user_coord = [lng, lat]
			#self.db.sendGPS(self.user_coord)
			return [lng,lat]
		
	
	def setupGSM(self):	
		print('initiate GSM')
		PORT = '/dev/ttyAMA0'
		BAUDRATE = 115200
		self.modem = GsmModem(PORT, BAUDRATE, incomingCallCallbackFunc=self.handleIncomingCall)
		self.modem.connect(None)
		print('Waiting for network coverage...')
		self.modem.waitForNetworkCoverage(30)
	
	def stopGSM(self):
		print('Stop Gsm Session')
		self.model.close()

	#To make phone call
	def callPhone(self, Number):
		if Number == None or Number == '00000':
			self.voice_assistant.speak('enter a valid number')
			return
		self.stopGPS()
		self.setupGSM()
		print('Dialing number: {0}'.format(Number))
		self.call = self.modem.dial(Number)
		print('Waiting for call to be answered/rejected')
		wasAnswered = False
		while self.call.active:
			if self.call.answered:
				wasAnswered = True
				if self.call.active: 
					self.voice_assistant.speak('call is attended')
				else:
					wasAnswered = False
					self.voice_assistant.speak('Call has been ended by remote party')
		if not wasAnswered:
			self.voice_assistant.speak('Call was not answered by remote party')
		self.modem.close()
		self.setUpGPS()
	
	def hangPhone(self):
		if (self.call):
			self.call.hangup()
		
	#To send SMS
	def sendSms(self, Number, Message):
		self.stopGPS()
		self.setupGSM()
		response = self.modem.sendSms(Number, Message, True)
		if type(response) == SentSms:
			self.voice_assistant.speak('SMS Delivered.')
		else:
			self.voice_assistant.speak('SMS Could not be sent')
		self.modem.close()
		self.setUpGPS()
		
	def handleIncomingCall(self,call):
		
		if call.ringCount == 1:
			self.voice_assistant.speak('Incoming call from:', call.number)
		elif call.ringCount >= 2:
			call.answer()
		else:
			self.voice_assistant.speak(' Call from {0} is still ringing...'.format(call.number))
		
	def setupGyro(self):
		print('Initiate GYRO')
		self.mpu = mpu6050(0x68)
		self.THRESHOLD = 10
		
	def listenGyro(self):
		accel_data = self.mpu.get_accel_data()
		acceleration = [accel_data['x'], accel_data['y'], accel_data['z']]
		magnitude = math.sqrt(sum([a ** 2 for a in acceleration]))
		if magnitude < self.THRESHOLD:
			self.fallDetected = True
			print("Fall detected!")
		else:
			self.fallDetected = False
	
	def setupVibrate(self):
		print('Initate vibration')
		vibration_pin = 17
		chip = gpiod.Chip('gpiochip4')
		self.led_line = chip.get_line(vibration_pin)
		self.led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)

	def vibrateMotor(self, t):
		try:
			self.led_line.set_value(1)
			time.sleep(t)
			self.led_line.set_value(0)
		finally:
			self.led_line.release()
	
	
#pi = piAPI('lol' , 'jh' )
#pi.vibrateMotor(1)
	
#def process():
	#while True:
		#time.sleep(0.05)
		#if (pi.GPSActive == True):
			#print(pi.getLocation())
		#pi.listenGyro()


#if __name__ == "__main__":
    #process()
   
