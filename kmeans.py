import sys, os, math, time
from PIL import Image, ImageFilter, ImageDraw
MAX_CLUSTERS = 50
MAX_ITERATIONS = 50
IMAGE_SIZE = []
EXTENSION = ".jpg"
CENTROIDS = {} 
NEW_CENTROIDS = {}
CLUSTER_NO = []
T0 = 0
T1 = 0
def getPalette(nameFile):

	global IMAGE_SIZE 
	global CENTROIDS

	dimEachColorPalette = 200
	image = 0
	try:
		image = Image.open(nameFile+EXTENSION)
	except IOError:
		print "IOError: open image. "
		sys.exit()

	IMAGE_SIZE = image.size
	
	print "size Original Image: " + str(IMAGE_SIZE)
	resize = int(0.5*IMAGE_SIZE[1])
    resizedImage = image.resize((resize, resize), Image.ANTIALIAS)
    result = resizedImage.convert('P', palette=Image.ADAPTIVE, colors=N_CLUSTERS)
	result.putalpha(0)
	colors = result.getcolors(resize*resize)
    pal = Image.new('RGB', (dimEachColorPalette*numColors, dimEachColorPalette))
	draw = ImageDraw.Draw(pal)

	posx = 0
	arrayColors = []
	
	for count, col in colors:
        draw.rectangle([posx, 0, posx+dimEachColorPalette, dimEachColorPalette], fill=col)
        arrayColors.append(col[:-1])
		posx = posx + dimEachColorPalette

 	del draw
 	
	CENTROIDS = dict(zip(range(0, N_CLUSTERS), arrayColors))

    
	pal.save(nameFile+"_cl"+str(N_CLUSTERS)+"_palette"+".png", "PNG")
	return image
def findCluster(pixel):
	dist = lambda pixel, clusterValue: math.sqrt( (clusterValue[1][0]-pixel[0])**2 + (clusterValue[1][1]-pixel[1])**2 + (clusterValue[1][2]-pixel[2])**2)
    centroidDistances = ([dist(pixel, cv) for cv in CENTROIDS.items()])
    myCluster = centroidDistances.index(min([x for x in centroidDistances ]))
    return myCluster

def findNewMean(image):

	global NEW_CENTROIDS


	count = [0] * N_CLUSTERS
	rgbValues = lambda: [0] * 3

	rgb4EachCluster = {key : rgbValues() for key in range(N_CLUSTERS)}

	for triple in CLUSTER_NO:
		count[triple[2]] += 1
		rgb4EachCluster[triple[2]][0] += image[triple[0],triple[1]][0]
		rgb4EachCluster[triple[2]][1] += image[triple[0],triple[1]][1]
		rgb4EachCluster[triple[2]][2] += image[triple[0],triple[1]][2]

	pos = 0

	for pos in range(N_CLUSTERS):
		value = [0,0,0]
		if count[pos] == 0:
			value[0] = rgb4EachCluster[pos][0]
			value[1] = rgb4EachCluster[pos][1]
			value[2] = rgb4EachCluster[pos][2]
		else:
			value[0] = rgb4EachCluster[pos][0]/count[pos]
			value[1] = rgb4EachCluster[pos][1]/count[pos]
			value[2] = rgb4EachCluster[pos][2]/count[pos]
		NEW_CENTROIDS[pos] = value
	

def swapOldNewCentroids():
	global CENTROIDS
	tempDict = {k:v for k,v in CENTROIDS.items()}
	CENTROIDS = {}
	CENTROIDS = {k:v for k,v in NEW_CENTROIDS.items()}
def saveNewImage(name):

	outputImage = Image.new("RGB",IMAGE_SIZE)
	print "Saving..."
	draw = ImageDraw.Draw(outputImage)

	for element in CLUSTER_NO:
		c = tuple(NEW_CENTROIDS.get(element[2]))
		draw.point((element[0],element[1]), fill=c)
	outputImage.save(name+"_cl"+str(N_CLUSTERS)+"_it"+str(N_ITERATIONS)+EXTENSION,"JPEG",quality=100)
	print "That's the end"
def kmeans(image):
	
	global CLUSTER_NO
	global NEW_CENTROIDS
	loadedImage = image.load()
	x = 0
	y = 0
	
	for iter in range(N_ITERATIONS):
		print "#Iteration : " + str(iter)
		CLUSTER_NO = []
		NEW_CENTROIDS = {}
		for x in range(IMAGE_SIZE[0]):
			for y in range(IMAGE_SIZE[1]):
				clusterTuple = [0] * 3
				clusterTuple[0] = x 
				clusterTuple[1] = y 
				clusterTuple[2] = findCluster(loadedImage[x,y]) 
				CLUSTER_NO.append(clusterTuple)

		findNewMean(loadedImage)
		swapOldNewCentroids()

def procedure(nameImage):

	image = getPalette(nameImage)
	T0=time.time()
	kmeans(image)
	T1=time.time()
	print "Time: " + str(T1-T0) + " seconds."
	saveNewImage(nameImage)
if __name__ == "__main__":

    global N_ITERATIONS
    global N_CLUSTERS
	args = sys.argv[1:]
   	inputFile = args[0]
    print "File Name: " + str(inputFile)
    try:
        numColors = int(args[1])
        iterations = int(args[2])
        N_ITERATIONS = iterations  
        N_CLUSTERS = numColors
        
    except: 
        print "Error: num Iterations"
        sys.exit()

    print "#Colors: " + str(numColors) + " #Iterations: " + str(iterations)

    nameFile,extension = os.path.splitext(inputFile)
    if (extension.lower() == ".jpg" or extension.lower() == ".jpeg") and numColors > 1 and numColors <= MAX_CLUSTERS and iterations > 0 and iterations <= MAX_ITERATIONS :
        procedure(nameFile)        
    else:
		print "Error parameters "
		sys.exit()