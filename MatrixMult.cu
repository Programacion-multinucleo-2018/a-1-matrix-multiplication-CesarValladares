#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "common.h"

using namespace std;

#define SIZEM 500;

void fillMatrices(float * ip, const int size){

    int i; 

    for (i = 0; i < size; i++){
        ip[i] = i;
    }    
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
    int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}

int main (int argc, char ** argv){

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    return 0;
}