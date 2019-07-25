import sys, os, math, time
import numpy as np
from PIL import Image

MAX_CLUSTERS = 50
IMAGE_SIZE = []
EXTENSION = ".jpg"




def getPalette(nameFile):

	global IMAGE_SIZE 
	global CENTROIDS
	dimEachColorPalette = 200
	image = 0
	try:
		image = Image.open(nameFile+EXTENSION)
	except IOError:
		print "Error open image"
		sys.exit()

	IMAGE_SIZE = image.size
	print "Size Original Image: " + str(IMAGE_SIZE)
	resize = int(0.5*IMAGE_SIZE[1])
    resizedImage = image.resize((resize, resize), Image.ANTIALIAS)
    result = resizedImage.convert('P', palette=Image.ADAPTIVE, colors=N_CLUSTERS)
	result.putalpha(0)
	colors = result.getcolors(resize*resize)	
	colors = [col[1][:-1] for col in colors]
	
	array = np.asarray(colors, dtype=np.uint8)
	print "Initial Centroids: "
	print array
	array.tofile(nameFile+"_palette"+str(N_CLUSTERS)+".raw")
	print nameFile+"_palette"+str(N_CLUSTERS)+".raw" + " saved."
if __name__ == "__main__":

    global N_CLUSTERS
    args = sys.argv[1:]
	inputFile = args[0]
    print "File Name: " + str(inputFile)
    try:
        N_CLUSTERS = int(args[1])
    except: 
        print "Error: #Initial Colors > 1"
        sys.exit()

    print "#Initial Centroids: " + str(N_CLUSTERS) 

    nameFile,extension = os.path.splitext(inputFile)
    if (extension.lower() == ".jpg" or extension.lower() == ".jpeg") and N_CLUSTERS > 1 and N_CLUSTERS <= MAX_CLUSTERS:
    	getPalette(nameFile)
    else:
		sys.exit()

