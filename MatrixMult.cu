#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "common.h"

using namespace std;

#define SIZEM 4;

void fillMatrices(float * ip, const int size){

    int i; 

    for (i = 0; i < size; i++){
        ip[i] = i;
    }    
}

__global__ void multMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
    int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}

void printM(float *ip){

    int size = SIZEM * SIZEM;
    for (int i = 0; u < size ; i++){
        printf (" %f ", ip[i]);
    }printf("\n");

}

int main (int argc, char ** argv){

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // Tamaño de la matriz
    int nx = SIZEM;
    int ny = SIZEM;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Apartar memoria 
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // Inicializar matrices
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Apartar memoria en la GPU
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // Transferir informacion a la GPU
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();

    duration_ms = end_cpu - start_cpu;

    printf("multMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return 0;
}