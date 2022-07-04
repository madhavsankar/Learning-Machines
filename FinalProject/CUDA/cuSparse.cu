#include <cusparse_v2.h>
#include <iostream>
#include <cuda_runtime.h>
#include "dnn.hpp"
#include <stdio.h>

#define N1 1000

// matrix generation and validation depends on these relationships:
#define SCL 2
#define K N1
#define M N1
// A: MxK  B: KxN  C: MxN

// error check macros
#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

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


// perform sparse-matrix multiplication C=AxB
int main(){

  cusparseStatus_t stat;
  cusparseHandle_t hndl;
  cusparseMatDescr_t descrA, descrB, descrC;
  int *csrRowPtrA, *csrRowPtrB, *csrRowPtrC, *csrColIndA, *csrColIndB, *csrColIndC;
  int *h_csrRowPtrA, *h_csrRowPtrB, *h_csrRowPtrC, *h_csrColIndA, *h_csrColIndB, *h_csrColIndC;
  float *csrValA, *csrValB, *csrValC, *h_csrValA, *h_csrValB, *h_csrValC;
  int nnzA, nnzB, nnzC;
  int m,n,k;
  m = M;
  n = N1;
  k = K;

// generate A, B=2I

/* A:
   |1.0 0.0 0.0 ...|
   |1.0 0.0 0.0 ...|
   |0.0 1.0 0.0 ...|
   |0.0 1.0 0.0 ...|
   |0.0 0.0 1.0 ...|
   |0.0 0.0 1.0 ...|
   ...

   B:
   |2.0 0.0 0.0 ...|
   |0.0 2.0 0.0 ...|
   |0.0 0.0 2.0 ...|
   ...                */
  double p_diag = 0.1;
  double p_nondiag = 0.1;
  //int NNZ;

  generateSquareSpMatrix(&h_csrValA, &h_csrRowPtrA, &h_csrColIndA, &nnzA, m, p_diag, p_nondiag);
  generateSquareSpMatrix(&h_csrValB, &h_csrRowPtrB, &h_csrColIndB, &nnzB, n, p_diag, p_nondiag);

// transfer data to device

  cudaMalloc(&csrRowPtrA, (m+1)*sizeof(int));
  cudaMalloc(&csrRowPtrB, (n+1)*sizeof(int));
  cudaMalloc(&csrColIndA, nnzA*sizeof(int));
  cudaMalloc(&csrColIndB, nnzB*sizeof(int));
  cudaMalloc(&csrValA, nnzA*sizeof(float));
  cudaMalloc(&csrValB, nnzB*sizeof(float));
  cudaCheckErrors("cudaMalloc fail");
  cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrB, h_csrRowPtrB, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, h_csrColIndA, nnzA*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndB, h_csrColIndB, nnzB*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrValA, h_csrValA, nnzA*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(csrValB, h_csrValB, nnzB*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy fail");

// set cusparse matrix types
  CUSPARSE_CHECK(cusparseCreate(&hndl));
  stat = cusparseCreateMatDescr(&descrA);
  CUSPARSE_CHECK(stat);
  stat = cusparseCreateMatDescr(&descrB);
  CUSPARSE_CHECK(stat);
  stat = cusparseCreateMatDescr(&descrC);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSE_CHECK(stat);
  stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSE_CHECK(stat);
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

// figure out size of C
  int baseC;
// nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnzC;
  stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
  CUSPARSE_CHECK(stat);
  cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
  cudaCheckErrors("cudaMalloc fail");
  stat = cusparseXcsrgemmNnz(hndl, transA, transB, m, n, k,
        descrA, nnzA, csrRowPtrA, csrColIndA,
        descrB, nnzB, csrRowPtrB, csrColIndB,
        descrC, csrRowPtrC, nnzTotalDevHostPtr );
  CUSPARSE_CHECK(stat);
  if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;}
  else{
    cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy fail");
    nnzC -= baseC;}
  cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
  cudaMalloc((void**)&csrValC, sizeof(float)*nnzC);
  cudaCheckErrors("cudaMalloc fail");
// perform multiplication C = A*B
  begin_roi();
  stat = cusparseScsrgemm(hndl, transA, transB, m, n, k,
        descrA, nnzA,
        csrValA, csrRowPtrA, csrColIndA,
        descrB, nnzB,
        csrValB, csrRowPtrB, csrColIndB,
        descrC,
        csrValC, csrRowPtrC, csrColIndC);
  end_roi();
  CUSPARSE_CHECK(stat);

// copy result (C) back to host
  h_csrRowPtrC = (int *)malloc((m+1)*sizeof(int));
  h_csrColIndC = (int *)malloc(nnzC *sizeof(int));
  h_csrValC  = (float *)malloc(nnzC *sizeof(float));
  if ((h_csrRowPtrC == NULL) || (h_csrColIndC == NULL) || (h_csrValC == NULL))
    {printf("malloc fail\n"); return -1;}
  cudaMemcpy(h_csrRowPtrC, csrRowPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrColIndC, csrColIndC,  nnzC*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_csrValC, csrValC, nnzC*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy fail");

  return 0;

}