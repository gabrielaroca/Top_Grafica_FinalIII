import sys, os, math, time
import numpy as np
from PIL import Image

W = 0
H = 0
global NAMEFILE
global EXTENSION
global RAW


def fromJpgToUint8():
	global W,H
	img = Image.open(NAMEFILE+EXTENSION)
	W = img.size[0]
	H = img.size[1]
	print "Width: " + str(W) + " Height: " + str(H)


	array = np.asarray(img)

	print "Saving .raw ..."
	array.tofile(NAMEFILE+RAW)
	print "Saved!"
	

def fromUint8ToJpg():
	array = np.fromfile(NAMEFILE+EXTENSION, dtype = "uint8")
	outputImage = array.reshape(H,W,3)
	image = Image.fromarray(outputImage)
	print "Saving .jpg ...."
	image.save(NAMEFILE+"_PYoutput"+".jpg" ,quality=100)
	print "Saved!"
	

if __name__ == "__main__":
	global NAMEFILE, EXTENSION, RAW, W, H
	RAW = ".raw"
	args = sys.argv[1:]
	inputFile = args[0]
	print "File Name: " + str(inputFile)
	try:
		NAMEFILE,EXTENSION = os.path.splitext(inputFile)
		if  (EXTENSION.lower() == ".jpg" or EXTENSION.lower() == ".jpeg"):
			fromJpgToUint8()
		elif EXTENSION.lower() == RAW:
			W = int(args[1])
			H = int(args[2])
			fromUint8ToJpg()
		else:
			print "Error:image format."
	except IOError:
		print "Error load"

