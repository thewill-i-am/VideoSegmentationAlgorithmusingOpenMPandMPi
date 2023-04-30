#include <stdio.h>
using namespace std;
#include <iostream>
#include <string>
#include <omp.h>
#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sys/time.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
using namespace cv;


double startTime;

void timerStart(){
	struct timeval tod;
	gettimeofday(&tod, NULL);
	startTime = (double)tod.tv_sec + ((double)tod.tv_usec * 1.0e-6);
}

double timerStop(){
	struct timeval tod;
	gettimeofday(&tod, NULL);
	return ((double)tod.tv_sec + ((double)tod.tv_usec * 1.0e-6)) - startTime;
}

void extract_frames(const string &videoFilePath,vector<Mat>& frames){
	
	try{
		//open the video file
  	VideoCapture cap(videoFilePath); // open the video file
  	if(!cap.isOpened())  // check if we succeeded
  		CV_Error(CV_StsError, "Can not open Video file");
	
  	//cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
  	for(int frameNum = 0; frameNum < cap.get(CAP_PROP_FRAME_COUNT);frameNum++)
  	{
  		Mat frame, hsvImg;
  		cap >> frame; // get the next frame from video

      cvtColor(frame, hsvImg, COLOR_BGR2HSV);
  		frames.push_back(hsvImg);
  	}
  }
  catch( cv::Exception& e ){
    cerr << e.msg << endl;
    exit(1);
  }
	
}


void save_frames(vector<Mat>& frames, const string& outputDir){
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);
  

  int frameNumber = 0;
  for(auto elemento : frames)
  {
    frameNumber = frameNumber + 1;
    string filePath = outputDir + to_string(static_cast<long long>(frameNumber))+ ".jpg";
	  imwrite(filePath,elemento,compression_params);

  }

}

Mat calcHistogramas(Mat* frames){
    Mat img1, hist;

    hist = 0;

    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins,
                      s_bins};

    float h_ranges[] = {0, 255};
    float s_ranges[] = {0, 255};

    const float *ranges[] = {h_ranges, s_ranges};

    int channels[] = {0, 1};

    img1 = frames[0];

    cv::calcHist(&img1, 1, channels, cv::Mat(), hist, 2, histSize, ranges);

    cv::normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );

    return hist;
}


void encontrarCorte(double *distanciasParciales, int numframes, char * outputFile){
  ofstream myfile (outputFile);

  for (size_t i = 0; i < numframes - 1; i++)
  {
    if(distanciasParciales[i] <  1.0 && distanciasParciales[i] != 0.0){
      // cout << "Frame " << i << ": " << distanciasParciales[i] << endl;
      myfile << "El corte esta en : " << i+1 << endl;
    }
  }

  myfile.close();
}

int main (int argc, char *argv[])
{

  char * outputFile = argv[2];

  vector<Mat> frames;
  
  //extract_frames("/media/william/ADATA HD680/prueba4.mp4",frames);
  //save_frames(frames, "/media/william/ADATA HD680/frames2/");

  int myRank;
  int p;
  double elapsedTime;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  cout << "Process " << myRank << " of " << p << " is running" << endl;

  double* distanciasTotales;
  
  VideoCapture cap(argv[1]); // open the video file
  
  if(!cap.isOpened()){
    return -1;
  }

  int numFrames = cap.get(CAP_PROP_FRAME_COUNT);

  int count = numFrames / p;
  int remainder = numFrames % p;
  int star, stop;


  /*Aqui se distribuyen los frames entre los ranks -> ING.Diego Jimenez nos ayudo con esta distribucion*/

  int *countGather = (int*)malloc(p*sizeof(int));
  int *displacementGather = (int*)malloc(p*sizeof(int));

  int sizeDisplacement = 0;

  for(int i = 0; i<p; i++){
    if(myRank < remainder){
      star = myRank * (count + 1);
      stop = star + count;
      countGather[i] = count + 1;
    }else{
      star = myRank * count + remainder;
      stop = star + count - 1;
      countGather[i] = count - 1;
    }
    displacementGather[i] = sizeDisplacement;
    sizeDisplacement += countGather[i];
  }

  /*Empieza el conteo*/
  if(myRank == 0){
    distanciasTotales = new double[numFrames];
    timerStart();
  }

  int sizePartial;

  if (myRank != p-1)
  {
    sizePartial = (stop - star) + 2;
  }else{
    sizePartial = (stop - star) + 1;
  }

  Mat* framesParciales = new Mat[sizePartial];
  
  int i;

  Mat* histogramasParciales = new Mat[sizePartial];

  double* distanciasParciales = new double[sizePartial];


  for (i = 0; i < sizePartial; i++)
  {
    cap.set(CAP_PROP_POS_FRAMES, star + i);
    cap >> framesParciales[i];
  }
  

  /*Obteniendo los histogramas */
  #pragma omp parallel for
  for (i = 0; i < sizePartial; i++)
  {
    Mat HSV;
    Mat RGB = framesParciales[i];
    cvtColor(RGB, HSV, COLOR_BGR2HSV);
    Mat res = calcHistogramas(&HSV);
    histogramasParciales[i] = res;
  }

  
  /*Calculo de la comparacion de los histogramas*/
  #pragma omp parallel for schedule(static, 2)
  for (int i = 0; i < sizePartial-1; i++)
  {
    distanciasParciales[i] =  -log(compareHist(histogramasParciales[i], histogramasParciales[i+1], HISTCMP_BHATTACHARYYA));
  }

   /*Esto ordena las respuestas de los procesos*/
   MPI_Gatherv(distanciasParciales, sizePartial - 1, MPI_DOUBLE, distanciasTotales, countGather, displacementGather, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   cout << "Process " << myRank << " of " << p << " is done" << endl;

  if(myRank == 0){
    encontrarCorte(distanciasTotales, numFrames, outputFile);
    elapsedTime = timerStop();
    cout << "Duration " << elapsedTime << "Seconds" << endl;
  }

  MPI_Finalize();

  return 0;
} 

//For compile
//mpicxx -openmp main.cpp -o main `pkg-config --cflags --libs opencv4`
//mpiexec -n 6 ./main "/media/william/ADATA HD680/prueba4.mp4"  "/media/william/ADATA HD680/texto.txt"
