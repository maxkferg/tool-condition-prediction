import xml.etree.ElementTree as ET
import urllib2
import sys
import time
import re

def createNoneMatrix(rows, columns):
	matrix = []
	for i in range(0, rows):
		array = []
		for j in range(0, columns):
			array.append(None)
		matrix.append(array)
	return matrix

def stringSum(str):
	lst = str.split()
	lst = map(int, lst)
	return sum(lst)

def dataExtract(machine, calibrationFactor, outputFile, timeLimit): 
	# Basic initialization

	AgentAddress = 'http://machineshop.dyndns.biz:5000/current'
	#testing static xml file: AgentAddress = 'C:\Users\dhanajay\Desktop\current.xml'

	# machine = 1 # 1 refers to the right machine and 0 refers to the left machine
	# timeLimit = 10 # duration in minutes
	# storeRange = 100 # choose storage range in seconds: (intial buffer)
	# Agent Address Validation

	try:
		xml = urllib2.urlopen(AgentAddress)
	except urllib2.URLError:
		sys.exit("Invalid MTConnect Agent Address! Please connect device to internet.")

	# Set machine for XML parsing

	if machine == 1:
		k = 1
	else:
		k = 0

	# Initialize Variables

	# #position
	# x = createNoneMatrix(storeRange/2, 1)
	# y = createNoneMatrix(storeRange/2, 1)
	# z = createNoneMatrix(storeRange/2, 1)

	# #load
	# lx = createNoneMatrix(storeRange/2, 1)
	# ly = createNoneMatrix(storeRange/2, 1)
	# lz = createNoneMatrix(storeRange/2, 1)
	# ls = createNoneMatrix(storeRange/2, 1)

	#power
	# P1 = createNoneMatrix(storeRange/2, 1)
	# P2 = createNoneMatrix(storeRange/2, 1)
	# P3 = createNoneMatrix(storeRange/2, 1)
	# P = createNoneMatrix(storeRange/2, 1)
	powerSequence = createNoneMatrix(2, 1)

	#feed, speed, time
	# feed = createNoneMatrix(storeRange/2, 1)
	# spindle = createNoneMatrix(storeRange/2, 1)
	# time1 = createNoneMatrix(storeRange/2, 1)
	# blockTimeStamp = createNoneMatrix(storeRange/2, 1)

	#block
	# block = createNoneMatrix(storeRange/2, 1)
	blockSequence = createNoneMatrix(2, 1)
	blockCount = 1
	# calibrationFactor = 0.046

	#loads
	sload = []
	xload = []
	yload = []
	zload = []
	LX = []
	LY = []
	LZ = []
	LS = []

	#powers
	p1 = []
	p2 = []
	p3 = []
	p = []

	#feedrate
	feed = []

	#speed
	speed = []

	#timestamp
	timestamp2 = []

	#duration
	duration = []

	#x,y,z
	X = []
	Y = []
	Z = []
	DX = []
	DY = []
	DZ = []

	#energy 
	energy = []

	#avg
	avgXload = []
	avgSload = []
	avgYload = []
	avgZload = []
	avgFeed = []
	avgSpeed = []

	#instruction
	instruction = []

	#read data
	# xml = urllib2.urlopen(AgentAddress)
	# xml = xml.read()
	# data = ET.ElementTree(ET.fromstring(xml))
	# test = data.getroot()
	# yolo = test.findall('.//')
	# partCount = []
	# for yolos in yolo:
	# 	if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}PartCount'):
	# 		partCount.append(yolos.text)

	#initialize time
	initial = time.time()
	timeout = time.time() + 60*timeLimit

	while time.time()<timeout:
		#progress = (time.time() - initial)/(60*timeLimit) * 100
		#print "Progress: ", progress
		execution = []
		xml = urllib2.urlopen(AgentAddress)
		xml = xml.read()
		data = ET.ElementTree(ET.fromstring(xml))
		test = data.getroot()
		yolo = test.findall('.//')
		for yolos in yolo:
			if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}Execution'):
				execution.append(yolos.text)
		if execution[k] == 'ACTIVE':
			#Block
			blockSeq = []
			blockData = []
			for yolos in yolo:
				if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}Block'):
					blockSeq.append(yolos.attrib)
					blockData.append(yolos.text)
			blockSequence[1][0] = blockSeq[k]['sequence']
			#print blockSequence
			#print (blockSequence[0][0] == blockSequence[1][0]) and blockCount > 1
			if (blockSequence[0][0] == blockSequence[1][0]) and blockCount > 1:
				#print "Im Here!"
				load = []
				for yolos in yolo:
					if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}Load'):
						load.append(yolos.text)
				LX.append(load[4*k + 1])
				LY.append(load[4*k + 2])
				LZ.append(load[4*k + 3])
				LS.append(load[4*k])
				# change 4*k for number machines being used
				#Real Power
				powerSeq = []
				powerData = []
				for yolos in yolo:
					if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}WattageTimeSeries'):
						powerSeq.append(yolos.attrib)
						powerData.append(yolos.text)
				powerSequence[1][0] = powerSeq[3*k+1]['sequence']
				if powerSequence[0][0] == powerSequence[1][0]:
					a = 0
				else:
					p1.append(stringSum(powerData[3*1]))
					p2.append(stringSum(powerData[3*1+1]))
					p3.append(stringSum(powerData[3*1+2]))
					p.append(p1[-1] + p2[-1] + p3[-1])
					powerSequence[0][0] = powerSequence[1][0]
				#path feedrate
				feedrate = []
				for yolos in yolo:
					if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}PathFeedrate'):
						feedrate.append(yolos.text)
				if k == 1:
					feed.append(feedrate[5])
				else:
					feed.append(feedrate[3*k+1])
				#spindle speed
				spindleSpeed = []
				for yolos in yolo:
					if (yolos.tag == '{urn:mtconnect.org:MTConnectStreams:1.3}SpindleSpeed'):
						spindleSpeed.append(yolos.text)
				speed.append(spindleSpeed[2*k])
			else:
				#timestamps
				text=blockSeq[1]['timestamp'];
				hour = 10* int(text[11]) + int(text[12])
				minute = 10* int(text[14]) + int(text[15])
				second = 10* int(text[17]) + int(text[18])
				thousandth = 100* int(text[20]) + 10*int(text[21])+int(text[22])
				blocktimestamp = []
				blocktimestamp.append(3600*hour + 60*minute + second + (float(1)/float(1000))*(thousandth))
				timestamp2.append(blocktimestamp[-1])

				#Get block durations
				if blockCount > 1:
					duration.append(float(timestamp2[-1])-float(timestamp2[-2]))
				#extract x, y, z positions
					if "G04" not in instruction[-1]:
						coords = re.split('([0-9]+\.[0-9]+|\d+|\-[0-9]+\.[0-9]+|\-\d+)', instruction[-1])
						coords = map(lambda str: str.replace(" ", ""), coords)
						xSeen = False
						ySeen = False
						zSeen = False
						for i in range(len(coords)):

							if (coords[i] == 'X' and xSeen == False):
								X.append(float(coords[i+1]))
								xSeen = True
							if (coords[i] == 'Y' and ySeen == False):
								Y.append(float(coords[i+1]))
								ySeen = True
							if (coords[i] == 'Z' and zSeen == False):
								Z.append(float(coords[i+1]))
								zSeen = True
						if (xSeen == False):
							X.append(float(X[-1]))
						if (ySeen == False):
							Y.append(float(Y[-1]))
						if (zSeen == False):
							Z.append(float(Z[-1]))
					else:
						X.append(float(X[-1]))
						Y.append(float(Y[-1]))
						Z.append(float(Z[-1]))
				else:
					X.append(0)
					Y.append(0)
					Z.append(0)
				#dx, dy, dz calc
				if blockCount > 1:
					DX.append(float(X[-1]) - float(X[-2]))
					DY.append(float(Y[-1]) - float(Y[-2]))
					DZ.append(float(Z[-1]) - float(Z[-2]))
				else:
					DX.append(0)
					DY.append(0)
					DZ.append(0)
				#concatenate data
				if blockCount == 1:
					datamax = [timestamp2[-1] - timestamp2[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, X[-1], Y[-1], Z[-1], 0, 0, 0]
				else:
						#energy calculation
					energy.append(sum(p)*0.01*calibrationFactor)
					#feed calc
					print "FEED:", feed
					if (len(feed) == 0):
						avgFeed.append(0)
					else:
						feed = map(float, feed)
						avgFeed.append(sum(feed)/len(feed))
					#feed speed
					if (len(speed) == 0):
						avgSpeed.append(0)
					else:
						speed = map(float, speed)
						avgSpeed.append(sum(speed)/len(speed))
					#loads calc
					if (len(LS) == 0):
						avgSload.append(0)
					else:
						LS = map(float, LS)
						avgSload.append(sum(LS)/len(LS))
					if (len(LX) == 0):
						avgXload.append(0)
					else:
						LX = map(float, LX)
						avgXload.append(sum(LX)/len(LX))
					if (len(LY) == 0):
						avgYload.append(0)
					else:
						LY = map(float, LY)
						avgYload.append(sum(LY)/len(LY))
					if (len(LZ) == 0):
						avgZload.append(0)
					else:
						LZ = map(float, LZ)
						avgZload.append(sum(LZ)/len(LZ))
					datamax = [timestamp2[-1] - timestamp2[0], instruction[-1].replace(" ", ""), duration[-1], energy[-1], avgFeed[-1], avgSpeed[-1], avgSload[-1], avgXload[-1], avgYload[-1], avgZload[-1], X[-1], Y[-1], Z[-1], DX[-1], DY[-1], DZ[-1]]
				#write to file
				f = open(outputFile+str(blockCount)+'.txt', 'w')
				dataToWrite = str(datamax).replace(",", "").replace("[", "").replace("]", "")
				f.write(dataToWrite)
				f.close()

				#reset averaging data
				speed = []
				feed = []
				p1 = []
				p2 = []
				p3 = []
				p = []
				LX = []
				LY = []
				LZ = []
				LS = []

				#print "datamax:", datamax
				#update block number
				blockCount = blockCount + 1
				#get new block
				instruction.append(blockData[k])
				blockSequence[0][0] = blockSequence[1][0]
				print "instruction", instruction[-1]


dataExtract(1, 0.0148, "./test/Block_", 10)
