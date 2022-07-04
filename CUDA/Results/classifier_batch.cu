#include <iostream>
#include <cuda_runtime.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 1024 // Number of Output Layers
  #define Ni 4096  // Number of Input  Layers
  #define BatchSize 16
#endif

void fill_classifier(float *synapse, float  *neuron_i, float *neuron_n) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n * Ni + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }

  for(int n = 0; n < Ni; ++n) {
    for(int i = 0; i < BatchSize; ++i) {
      neuron_i[n * BatchSize + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }

  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < BatchSize; ++i) {
      neuron_n[n * BatchSize + i] = 0;
    }
  }
}

void classifier_layer(float* synapse, float * neuron_i, float * neuron_n) {
  for (int n = 0; n < Nn; n++) 
    {
        for (int j = 0; j < BatchSize; ++j) 
        {
            float temp = 0;
            for (int i = 0; i < Ni; i++) 
            {
                temp += synapse[n * Ni + i] * neuron_i[i * BatchSize + j];
            }
            neuron_n[n * BatchSize + j] = transfer(temp);
        }
    }
}

__global__ void classifier_layer_batch_cuda(float* synapse, float* neuron_i, float* neuron_n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if( col < BatchSize && row < Nn) 
    {
        for(int i = 0; i < Ni; i++) 
        {
            sum += synapse[row * Ni + i] * neuron_i[i * BatchSize + col];
        }
        neuron_n[row * BatchSize + col] = sum;
    }
}

int main(int argc, char** argv) {

  float * synapse = (float *)malloc(Nn * Ni * sizeof(float));
  float * neuron_i = (float *)malloc(Ni * BatchSize * sizeof(float));
  float * neuron_n = (float *)malloc(Nn * BatchSize * sizeof(float));

  cudaError_t err = cudaSuccess;

  cout << "initializing arrays\n";

  fill_classifier(synapse,neuron_i,neuron_n);

  cout << "starting computation\n";

  begin_roi();
  classifier_layer(synapse,neuron_i,neuron_n);
  end_roi();

  cout << "simple version complete!\n";  


  // ===========================================================================
  // Allocate the device input vector A
  float *d_synapse = NULL;

  err = cudaMalloc((void **)&d_synapse, Nn * Ni * sizeof(float));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_neuron_i = NULL;
  err = cudaMalloc((void **)&d_neuron_i, Ni * BatchSize * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  // Allocate the device output vector C
  float *d_neuron_n = NULL;
  err = cudaMalloc((void **)&d_neuron_n, Nn * BatchSize * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // for(int i = 0; i < 10; i++)cout << synapse[i] << " ";
  // cout << endl;
  begin_roi();
  // Copy the host input vectors A and B in host memory to the device input vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_synapse, synapse, Nn * Ni * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_neuron_i, neuron_i, Ni * BatchSize * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 16;
  unsigned int grid_rows = (Nn + threadsPerBlock - 1) / threadsPerBlock;
  unsigned int grid_cols = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(threadsPerBlock, threadsPerBlock);

  
  //int blocksPerGrid =(Nn + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch\n");
  
  classifier_layer_batch_cuda<<<dimGrid, dimBlock>>>(d_synapse, d_neuron_i, d_neuron_n);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *h_C = (float *)malloc(Nn * BatchSize * sizeof(float));

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_neuron_n, Nn * BatchSize * sizeof(float), cudaMemcpyDeviceToHost);


  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


   // Free device global memory
   err = cudaFree(d_synapse);

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

   end_roi();

  // Free host memory
  free(synapse);
  free(neuron_i);
  free(neuron_n);

  printf("Done\n");

  return 0;
}

