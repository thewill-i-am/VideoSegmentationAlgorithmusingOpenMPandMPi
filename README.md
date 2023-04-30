# VideoSegmentation-AlgorithmusingOpenMPandMPi

This repository contains an efficient parallel algorithm for segmenting video into disjoint sets of homogeneous frames based on defined criteria. The algorithm combines two programming paradigms - shared memory with OpenMP and distributed memory with MPI - to achieve high performance on parallel architectures. The implementation includes two levels of parallelism and is designed to work with large video datasets. The code is written in C/C++ and includes examples demonstrating its use.

Parallel implementation of a video processing algorithm using MPI (Message Passing Interface) to distribute the workload across multiple processes.

The program first extracts frames from a video and saves them as images. Then, it calculates the histogram of the first frame using OpenCV's calcHist function, normalizes it and saves it as a reference histogram. Next, it calculates the histogram of each frame in the video and uses OpenCV's compareHist function to compare each histogram to the reference histogram. The resulting distances are stored in an array. Finally, it identifies the frames where a significant change occurs by looking for distances below a certain threshold and saves the results to an output file.

The MPI part of the code is used to distribute the workload across multiple processes. The program first reads the total number of frames in the video and calculates the number of frames that each process should process. It then uses MPI to distribute the frames among the processes. Each process calculates the distances for the frames it has been assigned and sends the results back to the root process. The root process then combines the results from all processes and identifies the frames where a significant change occurs. The results are saved to an output file.
