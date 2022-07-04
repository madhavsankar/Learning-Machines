#include <iostream>
#include <string>
#include "dnn.hpp"

using namespace std;

#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef A
  #define Nx_glob 224
  #define Ny_glob 224 
  #define Kx_glob 3
  #define Ky_glob 3 
  #define Ni_glob 64
  #define Nn_glob 64
  #define BatchSize_glob 16
#endif


void fill_convolution_shared_simple(float * synapse, 
                                    float* neuron_i,int BatchSize,int Ni,int Nn,int Nx,int Ny,int Kx,int Ky) {

      long NXPAD,NXSCL,NYSCL,NYPAD;
      NXPAD = Nx+Kx;
      NYPAD = Ny + Ky;
      NXSCL = Nx/Sx;
      NYSCL = Ny/Sy;                                
  
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          for(int yy = 0; yy < Ky; ++yy) {
            for(int xx = 0; xx < Kx; ++xx) {

          synapse[(nn*Ni+ni)*Ky*Kx + yy*Kx + xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
         
        } } } }

  for(int bbb=0;bbb<BatchSize;bbb++)
  {      
  for(int ni = 0; ni < Ni; ++ni) {
    for(int yy = 0; yy < NYPAD; ++yy) {
      for(int xx = 0; xx < NXPAD; ++xx) {      
      
        neuron_i[bbb*NXPAD*NXPAD*Ni + ni*NYPAD*NXPAD + yy*NXPAD + xx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
  }
}


void  convolution_layer(float * synapse, 
                               float * neuron_i, 
                               float * neuron_n2,int BatchSize,int Ni,int Nn,int Nx,int Ny,int Kx,int Ky) {

long NXPAD,NXSCL,NYSCL,NYPAD;
NXPAD = Nx+Kx;
NYPAD = Ny + Ky;
NXSCL = Nx/Sx;
NYSCL = Ny/Sy; 
for(int bbb = 0;bbb<BatchSize;bbb++)
{


for(int yyy=0;yyy<Ny;yyy+=Sy)
{
  for(int xxx=0;xxx<Nx;xxx+=Sx)
  {

  
for (int pp = 0;pp<Nn;pp += 1)
{
  float summer = 0.0;
  for(int z = 0;z<Ni;z += 1) {
  
        for (int ky = 0; ky < Ky; ky++)
        {
          for (int kx = 0; kx < Kx; kx++)
            
                {
                
                float sv = synapse[(pp*Ni+z)*Ky*Kx + ky*Kx + kx];
                
                float nv = neuron_i[bbb*NXPAD*NXPAD*Ni+ z*NXPAD*NYPAD + ((yyy)+ky)*NXPAD + (xxx)+kx];
           
                summer +=sv*nv;
             
                }
        }
    }
   
        neuron_n2[ bbb*NYSCL*NXSCL *Nn +  pp *NYSCL*NXSCL + (yyy/Sy)*NXSCL + (xxx/Sx)] = summer;
     
        
}
}
}
  }

                               }


__global__ void  convolution_layer_cuda(float * synapse, 
                               float * neuron_i, 
                               float * neuron_n,int BatchSize,int Ni,int Nn,int Nx,int Ny,int Kx,int Ky) {
 


long NXPAD,NXSCL,NYSCL,NYPAD;
NXPAD = Nx+Kx;
NYPAD = Ny + Ky;
NXSCL = Nx/Sx;
NYSCL = Ny/Sy; 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx<NXSCL*NYSCL)
  {
   
  int ycur = blockIdx.x*Sy ;
  int xcur = threadIdx.x*Sx ;
  int Nn_id = blockIdx.y;
  int batch_id = blockIdx.z;




  float summer = 0.0;
  for(int z = 0;z<Ni;z += 1) {
   
    
        

        for (int ky = 0; ky < Ky; ky++)
        {
          for (int kx = 0; kx < Kx; kx++)
            
                {
                float sv = synapse[(Nn_id*Ni+z)*Ky*Kx + ky*Kx + kx];
                float nv = neuron_i[batch_id*NXPAD*NYPAD*Ni + z*NXPAD*NYPAD + (ycur+ky)*NXPAD + xcur+kx];
                summer +=sv*nv;
            
                }
        }
    }
  
  
              
     
        neuron_n[batch_id*Nn*NYSCL*NXSCL + Nn_id *NYSCL*NXSCL + blockIdx.x*NXSCL + threadIdx.x] = summer;
       

      
}
  }
int main(int argc, char** argv) {

  long BatchSize;
  long Ni;
  long Nn;
  long Nx;
  long Ny;
  long Kx;
  long Ky;
  long Do_cpu;
  
  if(argc>1)
  {
    BatchSize = strtol(argv[1], NULL, 10);
    Ni = strtol(argv[2], NULL, 10);
    Nn = strtol(argv[3], NULL, 10);
    Nx = strtol(argv[4], NULL, 10);
    Ny = strtol(argv[5], NULL, 10);
    Kx = strtol(argv[6], NULL, 10);
    Ky = strtol(argv[7], NULL, 10);
    Do_cpu = strtol(argv[8], NULL, 10);

  }
  else
  {
    BatchSize = BatchSize_glob;
    Ni = Ni_glob;
    Nn = Nn_glob;
    Nx = Nx_glob;
    Ny = Ny_glob;
    Kx = Kx_glob;
    Ky = Ky_glob;
    Do_cpu = 0;
  }
  long NXPAD,NXSCL,NYSCL,NYPAD;
  NXPAD = Nx+Kx;
  NYPAD = Ny + Ky;
  NXSCL = Nx/Sx;
  NYSCL = Ny/Sy;


  
  cudaError_t err = cudaSuccess;
  float * synapse = (float *)malloc(Ky * Kx * Nn * Ni * sizeof(float));
  float * neuron_i = (float *)malloc(BatchSize* NYPAD * NXPAD * Ni * sizeof(float));
  float * neuron_n = (float *)malloc(BatchSize* NYSCL * NXSCL * Nn  * sizeof(float));
  float * neuron_n2 = (float *)malloc(BatchSize* NYSCL * NXSCL * Nn * sizeof(float));

  fill_convolution_shared_simple(synapse,neuron_i,BatchSize,Ni,Nn,Nx,Ny,Kx,Ky);

if(Do_cpu)
{
  begin_roi();
  convolution_layer(synapse,neuron_i,neuron_n2,BatchSize,Ni,Nn,Nx,Ny,Kx,Ky);
  end_roi();
} 


  float *d_synapse = NULL;

  err = cudaMalloc((void **)&d_synapse, Ky * Kx * Nn * Ni* sizeof(float));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  float *d_neuron_i = NULL;
  err = cudaMalloc((void **)&d_neuron_i,BatchSize * NYPAD * NXPAD * Ni * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  float *d_neuron_n = NULL;
  err = cudaMalloc((void **)&d_neuron_n,BatchSize* NYSCL * NXSCL * Nn * sizeof(float));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }



  

  err = cudaMemcpy(d_synapse, synapse, Ky * Kx * Nn * Ni * sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_neuron_i, neuron_i,BatchSize* NYPAD * NXPAD * Ni * sizeof(float), cudaMemcpyHostToDevice);
  

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  dim3 gridDim;
  gridDim.x = NYSCL;
  gridDim.y = Nn;
  gridDim.z = BatchSize;
  int threadsPerBlock = NXSCL;

  begin_roi();
  convolution_layer_cuda<<<gridDim, threadsPerBlock>>>(d_synapse, d_neuron_i, d_neuron_n,BatchSize,Ni,Nn,Nx,Ny,Kx,Ky);
  end_roi();
  printf("Convolution cuda baseline complete\n");
  err = cudaGetLastError();



  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch convo kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *h_C = (float *)malloc(BatchSize* NYSCL * NXSCL * Nn * sizeof(float));

  
  err = cudaMemcpy(h_C, d_neuron_n, BatchSize* NYSCL * NXSCL * Nn  * sizeof(float), cudaMemcpyDeviceToHost);


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

   

if(Do_cpu)
{
  compare(h_C,neuron_n2,BatchSize* Nn*NXSCL*NYSCL);
 
}
  free(synapse);
  free(neuron_i);
  free(neuron_n);
  free(neuron_n2);
  

  printf("Done\n");


  return 0;
  




}