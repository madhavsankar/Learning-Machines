#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 25088 // Number of Output Layers
  #define Ni 25088  // Number of Input  Layers
#endif

void fill_classifier(float *synapse, float  *neuron_i, float *neuron_n) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n * Ni + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0;
  }
}

void classifier_layer(float* synapse, float * neuron_i, float * neuron_n) {
  // int total_calc=0;
  for (int n = 0; n < Nn; n++) {
    float temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n * Ni + i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

float getRandomFloat(const float min, const float max)
{
        return ((((float)rand())/RAND_MAX)*(max-min)+min);
}

void resizeSpMatrixArraysAndCopy(float **A_temp_p, int **JA_temp_p, int *bufferSize_p, double RESIZE_FACTOR)
{

        printf("Executing resize!!\n");
        if (RESIZE_FACTOR <= 1) // RESIZE_FACTOR should not be less than one!
                RESIZE_FACTOR = 1.33; // if so, set to default value of 1.33

        int oldLength = (*bufferSize_p);
        int newLength = (int)ceil((*bufferSize_p)*RESIZE_FACTOR);
        float *A_temp_new;
        int *JA_temp_new;

        // allocate the new resized memory
        A_temp_new = (float *)malloc(sizeof(float)*newLength);
        JA_temp_new = (int *)malloc(sizeof(int)*newLength);

        // copy old elements into new array
        int i;
        for (i = 0; i < oldLength; ++i)
        {
                A_temp_new[i] = (*A_temp_p)[i];
                JA_temp_new[i] = (*JA_temp_p)[i];
        }

        // free memory from old arrays
        free(*A_temp_p);
        free(*JA_temp_p);

        // update pointers
        *A_temp_p = A_temp_new; A_temp_new = NULL;
        *JA_temp_p = JA_temp_new; A_temp_new = NULL;

        // update bufferSize
        *bufferSize_p = newLength;
}

void generateSquareSpMatrix(float **A_p, int **IA_p, int **JA_p, int *NNZ_p, const int N, const double p_diag, const double p_nondiag)
{
        // estimate size of A, JA arrays because they vary between realization
        // but are same for a given realization
        int estSize = N*p_diag + N*(N-1)*p_nondiag;
        
        // allocate IA because size is fixed (size of IA = N + 1)
        *IA_p = (int *)malloc(sizeof(int)*(N+1));
        
        // define buffer space for undetermined arrays
        int bufferSize = (int)ceil(1.33*estSize);
        
        // allocate buffer*estSize for A & JA so we can probably fit everything in those
        float* A_temp = (float *)malloc(sizeof(float)*bufferSize);
        int* JA_temp = (int *)malloc(sizeof(float)*bufferSize);
        
        double randProb; float randNum;

        // Setup inital conditions for sparse matrix
        *NNZ_p = 0; (*IA_p)[0] = 0;

        int i,j;
        for (i = 0; i < N; ++i)
        {
                (*IA_p)[i+1] = (*IA_p)[i];
                
                for (j = 0; j < N; ++j)
                {
                        randProb = ((double)rand())/RAND_MAX;
                        if (i == j) // on diagonal - use p_diag
                        {
                                if (randProb < p_diag) // insert non-zero element
                                {
                                        if((*NNZ_p) == bufferSize) // Placing element will exceed allowed buffer!
                                        {
                                                resizeSpMatrixArraysAndCopy(&A_temp, &JA_temp, &bufferSize, 1.33); // resize arrays so we can insert element!
                                        }
                                        
                                        // Place random non-zero element into sparse matrix
                                        randNum = getRandomFloat(0, 1);
                                        A_temp[(*NNZ_p)] = randNum;
                                        JA_temp[(*NNZ_p)] = j;
                                        (*IA_p)[i+1]++;
                                        (*NNZ_p)++;
                                }
                        }
                        else // not on diagonal - use p_nondiag
                        {
                                if (randProb < p_nondiag)
                                {
                                        if((*NNZ_p) == bufferSize) // Placing element will exceed allowed buffer!
                                        {
                                                resizeSpMatrixArraysAndCopy(&A_temp, &JA_temp, &bufferSize, 1.33); // resize arrays so we can insert element!
                                        }
                                        
                                        // Place random non-zero element into sparse matrix
                                        randNum = getRandomFloat(0, 1);;
                                        A_temp[(*NNZ_p)] = randNum;
                                        JA_temp[(*NNZ_p)] = j;
                                        (*IA_p)[i+1]++;
                                        (*NNZ_p)++;
                                        
                                }
                        }
                }
        }

        // By this point we have not exceeded memory limit so lets create
        // actual A and IA array now that we have determined the size
        *A_p = (float *)malloc(sizeof(float)*(*NNZ_p));
        *JA_p = (int *)malloc(sizeof(float)*(*NNZ_p));
        
        // Add elements from temp arrays to actual arrays
        for (i = 0; i < (*NNZ_p); ++i)
        {
                (*A_p)[i] = A_temp[i];
                (*JA_p)[i] = JA_temp[i];
        }
        
        // free no longer used temp arrays
        free(A_temp); A_temp = NULL;
        free(JA_temp); JA_temp = NULL;
        
        return;
}

__global__ void classifier_layer_cuda(float* matrix, float* x, float* y, int* row_ptr, int* col_ids)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float suma = 0.0;
    if (idx < Nn){
      const int row_start = row_ptr[idx];
      const int row_end = row_ptr[idx + 1];

      for(int j = row_start; j < row_end; j++)
        suma += matrix[j] * x[col_ids[j]];
      y[idx] = suma;
    }
}

int main(int argc, char** argv) {

  float * synapse = (float *)malloc(Nn * Ni * sizeof(float));
  float * neuron_i = (float *)malloc(Ni * sizeof(float));
  float * neuron_n = (float *)malloc(Nn * sizeof(float));
  float *A_cpu, *A_gpu;
  int *IA_cpu, *IA_gpu, *JA_cpu, *JA_gpu;
  //int *h_A_RowIndices = (int *)malloc((N + 1) * sizeof(*h_A_RowIndices));
  //int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));

  cudaError_t err = cudaSuccess;

  cout << "initializing arrays\n";

  fill_classifier(synapse,neuron_i,neuron_n);
  double p_diag = 0.7;
  double p_nondiag = 0.3;
  int NNZ;

  generateSquareSpMatrix(&A_cpu, &IA_cpu, &JA_cpu, &NNZ, Nn, p_diag, p_nondiag);

  cout << "starting computation\n";

  begin_roi();
  classifier_layer(synapse,neuron_i,neuron_n);
  end_roi();

  cout << "simple version complete!\n";  


  // ===========================================================================
  // Allocate the device input vector A
  // float *d_synapse = NULL;

  // err = cudaMalloc((void **)&d_synapse, Nn * Ni * sizeof(float));
  // if (err != cudaSuccess)
  // {
  //     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
  //     exit(EXIT_FAILURE);
  // }

  // // Allocate the device input vector B
  float *d_neuron_i = NULL;
  err = cudaMalloc((void **)&d_neuron_i, Ni * sizeof(float));

  // if (err != cudaSuccess)
  // {
  //    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
  //    exit(EXIT_FAILURE);
  // }
  // Allocate the device output vector C
  float *d_neuron_n = NULL;
  err = cudaMalloc((void **)&d_neuron_n, Nn * sizeof(float));

  // if (err != cudaSuccess)
  // {
  //     fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
  //     exit(EXIT_FAILURE);
  // }

  cudaMalloc((void**) &A_gpu, NNZ*sizeof(float));
  cudaMalloc((void**) &IA_gpu, (Nn+1)*sizeof(int)); // N = M
  cudaMalloc((void**) &JA_gpu, NNZ*sizeof(int));

  // for(int i = 0; i < 10; i++)cout << synapse[i] << " ";
  // cout << endl;
  begin_roi();
  // Copy the host input vectors A and B in host memory to the device input vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  // err = cudaMemcpy(d_synapse, synapse, Nn * Ni * sizeof(float), cudaMemcpyHostToDevice);

  // if (err != cudaSuccess)
  // {
  //     fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
  //     exit(EXIT_FAILURE);
  // }

  err = cudaMemcpy(d_neuron_i, neuron_i, Ni * sizeof(float), cudaMemcpyHostToDevice);

  // if (err != cudaSuccess)
  // {
  //     fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
  //     exit(EXIT_FAILURE);
  // }

  // Transfer to device
  cudaMemcpy(A_gpu, A_cpu, NNZ*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(IA_gpu, IA_cpu, (Nn+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(JA_gpu, JA_cpu, NNZ*sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(x_gpu, x_cpu, Nn*sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 16;
  int blocksPerGrid =(Nn + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  
  classifier_layer_cuda<<<blocksPerGrid, threadsPerBlock>>>(A_gpu, d_neuron_i, d_neuron_n, IA_gpu, JA_gpu);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *h_C = (float *)malloc(Nn * sizeof(float));

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_neuron_n, Nn * sizeof(float), cudaMemcpyDeviceToHost);


  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


   // Free device global memory
   err = cudaFree(A_gpu);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   err = cudaFree(d_neuron_i);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }

   err = cudaFree(d_neuron_n);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
   cudaFree(IA_gpu);
   cudaFree(JA_gpu);

   end_roi();

  // Free host memory
  free(synapse);
  free(neuron_i);
  free(neuron_n);
  free(A_cpu);
  free(IA_cpu);
  free(JA_cpu);

  printf("Done\n");

  return 0;
}

