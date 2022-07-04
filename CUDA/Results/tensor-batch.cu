#include <iostream>
#include <cuda_runtime.h>
#include "dnn.hpp"

using namespace std;
#include <mma.h>
using namespace nvcuda;


#ifndef Nn
  #define Nn_glob 4096 
  #define Ni_glob 25088  
  #define BatchSize_glob 16
#endif

 

const int tensor_batch = 16;
const int tensor_out_dim = 16;
const int tensor_inp_dim = 16;

void fill_classifier(float *synapse, float  *neuron_i, float *neuron_n, int BatchSize, int Ni, int Nn) {
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

__global__ void precision (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}
float compute_error(float* a,float* b,float size)
{
  float error = 0.0;
  for(int i=0;i<size;i++)
  {
    error += (a[i]-b[i])*(a[i]-b[i]);
  }


  error = sqrt(error/size);
  return error;
}


void classifier_layer(float* synapse, float * neuron_i, float * neuron_n, int BatchSize, int Ni, int Nn) {
  for (int n = 0; n < Nn; n++) 
    {
        for (int j = 0; j < BatchSize; ++j) 
        {
            float temp = 0;
            for (int i = 0; i < Ni; i++) 
            {
                temp += synapse[n * Ni + i] * neuron_i[i * BatchSize + j];
            }
            neuron_n[n * BatchSize + j] = temp;
        }
    }
}

__global__ void classifier_layer_batch_cuda(float* synapse, float* neuron_i, float* neuron_n, int BatchSize, int Ni, int Nn)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if( col < BatchSize && row < Nn) 
    {
        for(int i = 0; i < Ni; i++) 
        {
            sum += synapse[row * Ni + i] * neuron_i[i * BatchSize + col];
        }
        neuron_n[row * BatchSize + col] = sum;
    }
}

__global__ void tensor_cores(half *neuron_i, half *synapse, float *neuron_n, int BatchSize, int Ni, int Nn) {

   wmma::fragment<wmma::matrix_a, tensor_batch, tensor_out_dim, tensor_inp_dim, half, wmma::col_major> temp_a;
   wmma::fragment<wmma::matrix_b, tensor_batch, tensor_out_dim, tensor_inp_dim, half, wmma::col_major> temp_b;
   wmma::fragment<wmma::accumulator, tensor_batch, tensor_out_dim, tensor_inp_dim, float> c_temp;
   wmma::fill_fragment(c_temp, 0.0f);

   int tile_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int tile_y = (blockIdx.y * blockDim.y + threadIdx.y);
   for (int i = 0; i < Ni; i += tensor_inp_dim) {
         wmma::load_matrix_sync(temp_a, neuron_i + (tile_x * tensor_batch) + i * BatchSize, BatchSize);
         wmma::load_matrix_sync(temp_b, synapse + i + (tile_y * tensor_out_dim * Ni), Ni);
         wmma::mma_sync(c_temp, temp_a, temp_b, c_temp);
   }
  wmma::store_matrix_sync(neuron_n + (tile_x * tensor_batch) + (tile_y * tensor_out_dim) * BatchSize, c_temp, BatchSize, wmma::mem_col_major); 
}


int main(int argc, char** argv) {
  long BatchSize;
  long Ni;
  long Nn;
  long Do_cpu;
  if(argc>1)
  {
    BatchSize = strtol(argv[1], NULL, 10);
    Ni = strtol(argv[2], NULL, 10);
    Nn = strtol(argv[3], NULL, 10);
    Do_cpu = strtol(argv[4], NULL, 10);
  }
  else
  {
    BatchSize = BatchSize_glob;
    Ni = Ni_glob;
    Nn = Nn_glob;
    Do_cpu = 0;
  }

  float * synapse = (float *)malloc(Nn * Ni * sizeof(float));
  float * neuron_i = (float *)malloc(Ni * BatchSize * sizeof(float));
  float * neuron_n = (float *)malloc(Nn * BatchSize * sizeof(float));

  cudaError_t err = cudaSuccess;
 


  fill_classifier(synapse,neuron_i,neuron_n,BatchSize,Ni,Nn);

  if(Do_cpu)
  {

  begin_roi();
  classifier_layer(synapse,neuron_i,neuron_n,BatchSize,Ni,Nn);
  end_roi();
  cout << "simple version on CPU complete!\n\n"; 
  }

   


  float *d_synapse = NULL;

  err = cudaMalloc((void **)&d_synapse, Nn * Ni * sizeof(float));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector synapse (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  float *d_neuron_i = NULL;
  err = cudaMalloc((void **)&d_neuron_i, Ni * BatchSize * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector activation (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  float *d_neuron_n = NULL;
  err = cudaMalloc((void **)&d_neuron_n, Nn * BatchSize * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector d_neuron_n (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  float *c_tens = NULL;
  err = cudaMalloc((void **)&c_tens, Nn * BatchSize * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector c_tens (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }




  err = cudaMemcpy(d_synapse, synapse, Nn * Ni * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector synapse from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_neuron_i, neuron_i, Ni * BatchSize * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector neuron_i from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }



  half *a_fp16 ;
  half *b_fp16 ;

  cudaMalloc((void**)&a_fp16, BatchSize * Ni * sizeof(half));
  cudaMalloc((void**)&b_fp16, Ni * Nn * sizeof(half));

  dim3 gridDim;
  dim3 blockDim;
  blockDim.x = 32;
  blockDim.y = 32;
  int warpS = 32;
  gridDim.x = (BatchSize + ((tensor_batch * blockDim.x / warpS) - 1)) / (tensor_batch * blockDim.x / warpS);
  gridDim.y = (Nn + (tensor_out_dim * blockDim.y) - 1) / (tensor_out_dim * blockDim.y);
  

  int threadsPerBlock_tensor = 512;
  precision <<< (BatchSize * Ni + threadsPerBlock_tensor-1) / threadsPerBlock_tensor, threadsPerBlock_tensor >>> (a_fp16, d_neuron_i, BatchSize * Ni);
  precision <<< (Ni * Nn + threadsPerBlock_tensor-1) / threadsPerBlock_tensor, threadsPerBlock_tensor >>> (b_fp16, d_synapse, Ni * Nn);
  if(BatchSize>=16 and BatchSize%16==0)
  {


  begin_roi();
  tensor_cores <<< gridDim, blockDim >>> ( a_fp16, b_fp16, c_tens, BatchSize, Ni, Nn);
  end_roi();
  printf("Tensor core version complete\n\n");

  }
  
  
  float *h_C2 = (float *)malloc(BatchSize * Nn * sizeof(float));

  err = cudaMemcpy(h_C2, c_tens, BatchSize * Nn * sizeof(float), cudaMemcpyDeviceToHost);


  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  err = cudaFree(c_tens);

   if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }


  int threadsPerBlock = 16;
  unsigned int grid_rows = (Nn + threadsPerBlock - 1) / threadsPerBlock;
  unsigned int grid_cols = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(threadsPerBlock, threadsPerBlock);
  
  begin_roi();
  classifier_layer_batch_cuda<<<dimGrid, dimBlock>>>(d_synapse, d_neuron_i, d_neuron_n, BatchSize, Ni, Nn);
  end_roi();
  printf("Baseline version complete\n\n");

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch classifier kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *h_C_batch = (float *)malloc(Nn * BatchSize * sizeof(float));

 
  err = cudaMemcpy(h_C_batch, d_neuron_n, Nn * BatchSize * sizeof(float), cudaMemcpyDeviceToHost);
  if(BatchSize>=16 and BatchSize%16==0)
  { float error = 0.0;
    // printf("Comparing output of baseline vs tensor core\n");
    // compare(h_C_batch,h_C2,Nn*BatchSize);
    error =compute_error(h_C_batch,h_C2,Nn*BatchSize);
    cout<<"Total error is "<<error<<"\n";

    
  }
  if(Do_cpu)
  {

  printf("Comparing output of baseline vs CPU run\n");
  compare(h_C_batch,neuron_n,Nn*BatchSize);
  }

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


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



  free(synapse);
  free(neuron_i);
  free(neuron_n);
  free(h_C2);
  free(h_C_batch);

  printf("Done\n");

  return 0;
}

