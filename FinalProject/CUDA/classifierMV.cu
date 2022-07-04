#include <iostream>
#include <cuda_runtime.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 25088 // Number of Output Layers
  #define Ni 25088  // Number of Input  Layers
#endif

float compute_error(float* a, float* b, float size)
{
  float error = 0.0;
  for(int i=0;i<Nn;i++)
  {
    error += (a[i]-b[i])*(a[i]-b[i]);
  }

  error = sqrt(error/size);
  return error;
}

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
    float temp=0.0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n * Ni + i] * neuron_i[i];
    }

    neuron_n[n] = temp;
  }
}

__global__ void classifier_layer_cuda(float* synapse, float* neuron_i, float* neuron_n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float suma = 0.0;
    if (idx < Nn){
      for(int j = 0; j < Ni; j++)
      {
        suma += synapse[idx*Ni + j] * neuron_i[j];
      }

      neuron_n[idx] = suma;
    }
}

int main(int argc, char** argv) {

  float * synapse = (float *)malloc(Nn * Ni * sizeof(float));
  float * neuron_i = (float *)malloc(Ni * sizeof(float));
  float * neuron_n = (float *)malloc(Nn * sizeof(float));

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
  err = cudaMalloc((void **)&d_neuron_i, Ni * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  // Allocate the device output vector C
  float *d_neuron_n = NULL;
  err = cudaMalloc((void **)&d_neuron_n, Nn * sizeof(float));

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

  err = cudaMemcpy(d_neuron_i, neuron_i, Ni * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 16;
  int blocksPerGrid =(Nn + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  
  classifier_layer_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_synapse, d_neuron_i, d_neuron_n);

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

  float error = 0.0;
  for(int i = 0; i < Nn; i++)
  {
    error += (h_C[i]-neuron_n[i])*(h_C[i]-neuron_n[i]);
    //cout<<h_C[i]<<" "<<neuron_n[i]<<"\n";
  }

  error = sqrt(error/Nn);
  cout<<"Total error is "<<error<<"\n";

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

