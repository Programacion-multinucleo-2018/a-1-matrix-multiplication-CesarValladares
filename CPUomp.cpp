#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "omp.h"

using namespace std;

#define SIZEM 500;

void fillMatrices(float * ip, const int size){

    int i; 

    for (i = 0; i < size; i++){
        ip[i] = i;
    }    
}

void MultFuncion(float * h_A, float * h_B, float * hostRef, int nx, int ny){

    //Mult(h_A, h_B, hostRef, nx);
    #pragma omp parallel for private(i) shared(h_A,h_B,hostRef)

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            float sum = 0.0;
            for (int k = 0; k < ny; k++)
                sum = sum + h_A[i * nx + k] * h_B[k * nx + j];
            hostRef[i * nx + j] = sum;
        }
    }
}

int main(){

    //informacion del tamaño de la matriz
    int nx = SIZEM;
    int ny = SIZEM; 
    int nxy = nx * ny; 
    int nBytes = nxy * sizeof(int);

    //printf("Matrix size: nx %d ny %d\n", nx, ny);

    //malloc
    float *h_A, *h_B, *hostRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);

    //inicializar 
    fillMatrices(h_A, nxy);
    fillMatrices(h_B, nxy);

    int x = SIZEM;

    int repeticiones = 100;
    auto promedio = 0.0;

    for ( int i = 0 ; i < repeticiones ; i++){
        auto startTime = chrono::high_resolution_clock::now();
        MultFuncion(h_A, h_B, hostRef, x, x);
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = endTime - startTime;
        promedio += duration_ms.count();
    }
    
    promedio /= repeticiones;
    
    free(h_A);
    free(h_B);
    free(hostRef);

    printf("Promedio de tiempo CPU con omp en %d repeticiones:  %f ms con una matriz x: %d y: %d\n", repeticiones,promedio, x, x);

    return 0;
}