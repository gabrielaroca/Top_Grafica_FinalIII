#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "timer.h"

#define BLOCK_SIZE 16
#define GRID_SIZE 256

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}


__constant__ int dev_nCentroids;
__constant__ int dev_size;


int PALETTE_BYTES = 0;
int IMAGE_BYTES = 0;  

__constant__ int dev_RedCentroid[20];
__constant__ int dev_GreenCentroid[20];
__constant__ int dev_BlueCentroid[20];

//RGB Values
bool loadPalette(char* filename, int nCentroids, int* redCentroid, int* greenCentroid, int*  blueCentroid) {

	FILE *imageFile;
	int length = 0;

	imageFile = fopen(filename,"r");
	if (imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < nCentroids; i++) {

			
			redCentroid[i] = fgetc(imageFile);
			greenCentroid[i] = fgetc(imageFile);
			blueCentroid[i] = fgetc(imageFile);
			printf("%d, %d, %d\n",redCentroid[i], greenCentroid[i], blueCentroid[i] );
			length++;
		}
		fclose(imageFile);
		printf("\n");
		printf("Tamaño de la paleta: %d\n", length);
		return true;
	}
}


bool loadRawImage(char* filename, int* r, int* g, int* b, int size) {
	FILE *imageFile;
	imageFile = fopen(filename, "r");

	if (imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < size; i++) {

			r[i] = fgetc(imageFile);
			g[i] = fgetc(imageFile);
			b[i] = fgetc(imageFile);
		}
		fclose(imageFile);

		/*for(int j = 0; j < h * w; j++) {
			printf("%d, %d, %d ", r[j], g[j], b[j]);
		}*/
		return true;
	}
}

bool writeRawImage(char* filename, int* labelArray, int* redCentroid, int* greenCentroid, int* blueCentroid, int size){
	FILE *imageFile;
	imageFile = fopen(filename, "wb");

	if(imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < size; i++) {
			fputc((char) redCentroid[labelArray[i]], imageFile);
			fputc((char) greenCentroid[labelArray[i]], imageFile);
			fputc((char) blueCentroid[labelArray[i]], imageFile);
		}
		fclose(imageFile);
		return true;
	}
}

__global__ void clearPaletteArrays(int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue, int* dev_pixelClusterCounter, int* dev_tempRedCentroid, int* dev_tempGreenCentroid, int* dev_tempBlueCentroid ) {

	// 1 block, 16x16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids) {

		// nCentroids long
		dev_sumRed[threadID] = 0;
		dev_sumGreen[threadID] = 0;
		dev_sumBlue[threadID] = 0;
		dev_pixelClusterCounter[threadID] = 0;
		dev_tempRedCentroid[threadID] = 0;
		dev_tempGreenCentroid[threadID] = 0;
		dev_tempBlueCentroid[threadID] = 0;
	}
}


__global__ void clearLabelArray(int *dev_labelArray){

	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	if(threadID < dev_size) {
		dev_labelArray[threadID] = 0;
	}
}

__global__ void getClusterLabel(int *dev_Red,int *dev_Green,int *dev_Blue,int *dev_labelArray) {


	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	float min = 500.0, value;
	int index = 0;


	if(threadID < dev_size) {
		for(int i = 0; i < dev_nCentroids; i++) {
			value = sqrtf(powf((dev_Red[threadID]-dev_RedCentroid[i]),2.0) + powf((dev_Green[threadID]-dev_GreenCentroid[i]),2.0) + powf((dev_Blue[threadID]-dev_BlueCentroid[i]),2.0));

			if(value < min){
				
				min = value;
				
				index = i;
			}
		}
		dev_labelArray[threadID] = index;

	}
}

__global__ void sumCluster(int *dev_Red,int *dev_Green,int *dev_Blue,int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue,int *dev_labelArray,int *dev_pixelClusterCounter) {

	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;


	if(threadID < dev_size) {
		int currentLabelArray = dev_labelArray[threadID];
		int currentRed = dev_Red[threadID];
		int currentGreen = dev_Green[threadID];
		int currentBlue = dev_Blue[threadID];
		atomicAdd(&dev_sumRed[currentLabelArray], currentRed);
		atomicAdd(&dev_sumGreen[currentLabelArray], currentGreen);
		atomicAdd(&dev_sumBlue[currentLabelArray], currentBlue);
		atomicAdd(&dev_pixelClusterCounter[currentLabelArray], 1);
	}
}

__global__ void newCentroids(int *dev_tempRedCentroid, int *dev_tempGreenCentroid, int *dev_tempBlueCentroid,int* dev_sumRed, int *dev_sumGreen,int *dev_sumBlue, int* dev_pixelClusterCounter) {

	// 1 block , 16*16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids) {
		int currentPixelCounter = dev_pixelClusterCounter[threadID];
		int sumRed = dev_sumRed[threadID];
		int sumGreen = dev_sumGreen[threadID];
		int sumBlue = dev_sumBlue[threadID];

		//new RGB Centroids' values written in global memory
		dev_tempRedCentroid[threadID] = (int)(sumRed/currentPixelCounter);
		dev_tempGreenCentroid[threadID] = (int)(sumGreen/currentPixelCounter);
		dev_tempBlueCentroid[threadID] = (int)(sumBlue/currentPixelCounter);
	}

}

int main(int argc, char *argv[]) {

		// init device
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaThreadSynchronize();

		
		char *inputFile, *outputFile, *palette;
		int *red, *green, *blue, *redCentroid, *greenCentroid, *blueCentroid;
		int *dev_Red, *dev_Green, *dev_Blue, *dev_tempRedCentroid, *dev_tempGreenCentroid, *dev_tempBlueCentroid;
		int *labelArray, *dev_labelArray;
		int width, height, nCentroids, nIterations,size;
		int *pixelClusterCounter, *dev_pixelClusterCounter;
		int *sumRed, *sumGreen, *sumBlue;
		int *dev_sumRed, *dev_sumGreen, *dev_sumBlue;

		
		if (argc > 7) {
			inputFile = argv[1];
			outputFile = argv[2];
			width = atoi(argv[3]);
			height = atoi(argv[4]);
			palette = argv[5];
			nCentroids = atoi(argv[6]);  
			if(nCentroids > 256)
				nCentroids = 256;
			nIterations = atoi(argv[7]);
			if(nIterations > 15)
				nIterations = 15;

		} else {
			printf("  Compilar: kmeans.cu <inputfile.raw> <outputfile.raw> nRows nCols paleta nCentroides nItarationes \n");
			return 0;
		}

		
		IMAGE_BYTES = width * height * sizeof(int);
		PALETTE_BYTES = nCentroids * sizeof(int);
		size = width * height;


		printf("Image: %s\n",inputFile);
		printf("Width: %d, Height: %d\n", width, height);
		printf("#Clusters: %d, #Iterations: %d\n", nCentroids, nIterations);

		red = static_cast<int *>(malloc(IMAGE_BYTES));
		green = static_cast<int *>(malloc(IMAGE_BYTES));
		blue = static_cast<int *>(malloc(IMAGE_BYTES));
		redCentroid = static_cast<int *>(malloc(PALETTE_BYTES));
		greenCentroid = static_cast<int *>(malloc(PALETTE_BYTES));
		blueCentroid = static_cast<int *>(malloc(PALETTE_BYTES));
		labelArray = static_cast<int *>(malloc(IMAGE_BYTES));
		sumRed = static_cast<int*>(malloc(PALETTE_BYTES));
		sumGreen = static_cast<int*>(malloc(PALETTE_BYTES));
		sumBlue = static_cast<int*>(malloc(PALETTE_BYTES));
		pixelClusterCounter = static_cast<int*>(malloc(PALETTE_BYTES));

		printf("Initial Centroids: \n");
		if(loadPalette(palette, nCentroids, redCentroid, greenCentroid, blueCentroid)) {
		} else {
			printf("Unable to set Initial Centroids.\n");
		}

		printf("Image loading...\n");
		if (loadRawImage(inputFile, red, green, blue, size)) {
			printf("Image loaded!\n");
		} else {
			printf("NOT loaded!\n");
			return -1;
		}

		printf("\n");

		

		if(IMAGE_BYTES == 0 || PALETTE_BYTES == 0) {
			return -1;
		}

		CUDA_CALL(cudaMalloc((void**) &dev_Red, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_Green, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_Blue, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempRedCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempGreenCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempBlueCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_labelArray, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumRed, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumGreen, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumBlue, PALETTE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_pixelClusterCounter, PALETTE_BYTES));

		// copy host CPU memory to GPU
		CUDA_CALL(cudaMemcpy(dev_Red, red, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_Green, green, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_Blue, blue, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_tempRedCentroid, redCentroid,PALETTE_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_tempGreenCentroid, greenCentroid,PALETTE_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_tempBlueCentroid, blueCentroid,PALETTE_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_labelArray, labelArray, IMAGE_BYTES, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_sumRed, sumRed, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_sumGreen, sumGreen, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_sumBlue, sumBlue, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_pixelClusterCounter, pixelClusterCounter, PALETTE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, redCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, greenCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, blueCentroid, PALETTE_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_nCentroids,&nCentroids, sizeof(int)));
		CUDA_CALL(cudaMemcpyToSymbol(dev_size, &size, sizeof(int)));


		// Clearing centroids on host
		for(int i = 0; i < nCentroids; i++) {
			redCentroid[i] = 0;
			greenCentroid[i] = 0;
			blueCentroid[i] = 0;
		}

		// Defining grid size

		int BLOCK_X, BLOCK_Y;
		BLOCK_X = ceil(width/BLOCK_SIZE);
		BLOCK_Y = ceil(height/BLOCK_SIZE);
		if(BLOCK_X > GRID_SIZE)
			BLOCK_X = GRID_SIZE;
		if(BLOCK_Y > GRID_SIZE)
			BLOCK_Y = GRID_SIZE;

		//2D Grid
		//Minimum number of threads that can handle width¡height pixels
	 	dim3 dimGRID(BLOCK_X,BLOCK_Y);
	 	//2D Block
	 	//Each dimension is fixed
		dim3 dimBLOCK(BLOCK_SIZE,BLOCK_SIZE);

		//Starting timer
		GpuTimer timer;
		timer.Start();

		printf("Launching K-Means Kernels..	\n");
		//Iteration of kmeans algorithm
		for(int i = 0; i < nIterations; i++) {


			clearPaletteArrays<<<1, dimBLOCK>>>(dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter, dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid);

			clearLabelArray<<<dimGRID, dimBLOCK>>>(dev_labelArray);
			
			getClusterLabel<<< dimGRID, dimBLOCK >>> (dev_Red, dev_Green, dev_Blue,dev_labelArray);


			sumCluster<<<dimGRID, dimBLOCK>>> (dev_Red, dev_Green, dev_Blue, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_labelArray,dev_pixelClusterCounter);


			newCentroids<<<1,dimBLOCK >>>(dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter);


			CUDA_CALL(cudaMemcpy(redCentroid, dev_tempRedCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(greenCentroid, dev_tempGreenCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(blueCentroid, dev_tempBlueCentroid, PALETTE_BYTES,cudaMemcpyDeviceToHost));

			CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, redCentroid, PALETTE_BYTES));
			CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, greenCentroid, PALETTE_BYTES));
			CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, blueCentroid, PALETTE_BYTES));
			timer.Stop();
		}


		CUDA_CALL(cudaMemcpy(labelArray, dev_labelArray, IMAGE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumRed, dev_sumRed, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumGreen, dev_sumGreen, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumBlue, dev_sumBlue, PALETTE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(pixelClusterCounter, dev_pixelClusterCounter, PALETTE_BYTES, cudaMemcpyDeviceToHost));

		printf("Kmeans tiempo: %f msecs.\n", timer.Elapsed());
		printf("\n");

		
		  int counter = 0;

		printf("Label Array:\n");
		for(int i = 0; i < (size); i++) {
			//printf("%d\n", labelArray[i]);
			counter++;
		}
		printf("printing counter %d\n", counter);
		counter = 0;

		printf("Pallete:\n");
		for(int j = 0; j < nCentroids; j++) {
			printf("r: %u g: %u b: %u \n", sumRed[j], sumGreen[j], sumBlue[j]);
			counter++;
		}

		printf("\n");

		printf("Pixels por centroides:\n");
		for(int k = 0; k < nCentroids; k++){
			printf("%d centroid: %d pixels\n", k, pixelClusterCounter[k]);
		}

		printf("\n");



		printf("Nuevos centroides:\n");
		for(int i = 0; i < nCentroids; i++) {

			printf("%d, %d, %d \n", redCentroid[i], greenCentroid[i], blueCentroid[i]);
		}


		
		printf("Image ...\n");

		if (writeRawImage(outputFile,labelArray, redCentroid, greenCentroid,  blueCentroid,  size)) {
			printf("Image procesada \n");
		} else {
			printf("No procesada\n");
			return -1;
		}

		free(red);
		free(green);
		free(blue);
		free(redCentroid);
		free(greenCentroid);
		free(blueCentroid);
		free(labelArray);
		free(sumRed);
		free(sumGreen);
		free(sumBlue);
		free(pixelClusterCounter);

		CUDA_CALL(cudaFree(dev_Red));
		CUDA_CALL(cudaFree(dev_Green));
		CUDA_CALL(cudaFree(dev_Blue));
		CUDA_CALL(cudaFree(dev_tempRedCentroid));
		CUDA_CALL(cudaFree(dev_tempGreenCentroid));
		CUDA_CALL(cudaFree(dev_tempBlueCentroid));
		CUDA_CALL(cudaFree(dev_labelArray));
		CUDA_CALL(cudaFree(dev_sumRed));
		CUDA_CALL(cudaFree(dev_sumGreen));
		CUDA_CALL(cudaFree(dev_sumBlue));
		CUDA_CALL(cudaFree(dev_pixelClusterCounter));

		printf("Elapsed time...\n");
		return 0;
}