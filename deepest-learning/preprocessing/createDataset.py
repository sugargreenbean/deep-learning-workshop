import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

personName = input('Enter your full name as it appears on Facebook: ')

def getFacebookData():
	responseDictionary = dict()
	fbFile = open('./html/fbMessages.txt', 'r')
	allLines = fbFile.readlines()
	myMessage, otherPersonsMessage, currentSpeaker = "","",""
	for index,lines in enumerate(allLines):
	    rightBracket = lines.find(']') + 2
	    justMessage = lines[rightBracket:]
	    colon = justMessage.find(':')
	    # Find messages that I sent
	    if (justMessage[:colon] == personName):
	        if not myMessage:
	            # Want to find the first message that I send (if I send multiple in a row)
	            startMessageIndex = index - 1
	        myMessage += justMessage[colon+2:]

	    elif myMessage:
	        # Now go and see what message the other person sent by looking at previous messages
	        for counter in range(startMessageIndex, 0, -1):
	            currentLine = allLines[counter]
	            rightBracket = currentLine.find(']') + 2
	            justMessage = currentLine[rightBracket:]
	            colon = justMessage.find(':')
	            if not currentSpeaker:
	                # The first speaker not named me
	                currentSpeaker = justMessage[:colon]
	            elif (currentSpeaker != justMessage[:colon] and otherPersonsMessage):
	                # A different person started speaking, so now I know that the first person's message is done
	                otherPersonsMessage = cleanMessage(otherPersonsMessage)
	                myMessage = cleanMessage(myMessage)
	                responseDictionary[otherPersonsMessage] = myMessage
	                break
	            otherPersonsMessage = justMessage[colon+2:] + otherPersonsMessage
	        myMessage, otherPersonsMessage, currentSpeaker = "","",""
	return responseDictionary

def cleanMessage(message):
	# Remove new lines within message
	cleanedMessage = message.replace('\n',' ').lower()
	# Deal with some weird tokens
	cleanedMessage = cleanedMessage.replace("\xc2\xa0", "")
	# Remove punctuation
	cleanedMessage = re.sub('([.,!?])','', cleanedMessage)
	# Remove multiple spaces in message
	cleanedMessage = re.sub(' +',' ', cleanedMessage)
	return cleanedMessage

combinedDictionary = {}
combinedDictionary.update(getFacebookData())
print('Total len of dictionary', len(combinedDictionary))
print('Saving conversation data dictionary')
np.save('conversationDictionary.npy', combinedDictionary)

conversationFile = open('conversationData.txt', 'w')
for key,value in combinedDictionary.items():
	if (not key.strip() or not value.strip()):
		# If there are empty strings
		continue
	conversationFile.write(key.strip() + ' '+ value.strip() + ' ')
