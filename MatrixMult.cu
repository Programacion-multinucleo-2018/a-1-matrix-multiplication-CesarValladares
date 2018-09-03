#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include "common.h"

using namespace std;

#define SIZEM 1000;

void fillMatrices(int * ip, const int size){

    int i; 

    for (i = 0; i < size; i++){
        ip[i] = i;
    }    
}

__global__ void multMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx,
    int ny)
{   
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int idx = ix * nx + iy;

    int auxiliar = 0;

    if (ix < nx && iy < ny){
        for(int i = 0; i < ny ; i++){
            auxiliar += MatA[ix * nx + i] * MatB[i * ny + iy];
        }
    }

    MatC[idx] = auxiliar;
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
    int nBytes = nxy * sizeof(int);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Apartar memoria 
    int *h_A, *h_B, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    // Inicializar matrices
    fillMatrices(h_A, nxy);
    fillMatrices(h_B, nxy);

    memset(gpuRef, 0, nBytes);

    // Apartar memoria en la GPU
    int *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // Transferir informacion a la GPU
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // Invocar al kernel del lado del host
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    int repetitions = 30;
    auto average = 0;

    for (int i = 0; i < repetitions; i++){
        auto start_cpu =  chrono::high_resolution_clock::now();
        multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
        auto end_cpu =  chrono::high_resolution_clock::now();
        
        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
        average += duration_ms.count();
    }

    average /= repetitions;

    printf("multMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %d ms in %d repetitions\n", grid.x,
           grid.y,
           block.x, block.y, average, repetitions);

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return 0;
}