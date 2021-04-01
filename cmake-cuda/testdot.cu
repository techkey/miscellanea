#include <iostream>
#include <chrono>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cuda_runtime.h>
#include "cublas_v2.h"

int main(int argc, char **argv){

  if (argc != 3) {
    std::cout<<"Wrong number of arguments!!!"<<std::endl;
    std::cout<<"Usage "<<argv[0]<<" n NSample"<<std::endl;
    return -1;
  }

#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp master
  std::cout<<"Using "<<omp_get_num_threads() <<" threads out of a maximum of"<<omp_get_max_threads()<<std::endl;
#endif 

  int n=atoi(argv[1]);
  int nSamples=atoi(argv[2]);
  std::cout<<"Running... ";
  for(int i=0; i<argc; ++i) {
    std::cout<<argv[i]<<" ";
  }
  std::cout<<std::endl;


  double *times = new double[nSamples];
  double *A = new double[n];
  double *B = new double[n];
  double tmin=std::numeric_limits<double>::max();
  double tmax=0.0;
  double z;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  double* devPtrA;
  double* devPtrB;
    if (!A) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, n*sizeof(*A));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrB, n*sizeof(*B));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
for (int i=1; i<=nSamples; ++i) {
    for (int j=0; j<n; ++j) {
      A[j] = (double) i * (double)(j+1)/(double) n;
      B[j] = (double) i * ((double)j+1.5)/(double) n;
    //  std::cout << A[j] << " " << B[j] << std::endl;
    }
  
    stat = cublasSetVector(n, sizeof(double),
                A, 1, devPtrA , 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetVector(n, sizeof(double),
                B, 1, devPtrB , 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    const auto start = std::chrono::high_resolution_clock::now();
#ifdef WITH_BLAS
    z=cblas_ddot(n,A,1,B,1);
#else    
    stat = cublasDdot(handle, n, devPtrA, 1, devPtrB, 1, &z);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
//    z=dot(A,B,n);
#endif    
    times[i-1] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    if (tmin>times[i-1]) tmin=times[i-1];
    if (tmax<times[i-1]) tmax=times[i-1];

  }
  double timeAvg = 0.0;
  for(int i=0; i<nSamples; ++i) {
    timeAvg+=times[i];
  }
  timeAvg/=nSamples;

  double sig=0.0;
  for(int i=0; i<nSamples; ++i) {
    sig+=(times[i]-timeAvg)*(times[i]-timeAvg);
  }
  sig=sqrt(sig/nSamples);

  std::cout<<"#Loop  |       Size|        Time (s)|"<<std::endl;
  for(int i=0; i<nSamples; ++i) {
    std::cout<<i+1<<" "<<n<<" "<<times[i]<<std::endl;
  }
  std::cout<<"Last inner product: "<<z<<std::endl;
  std::cout<<"Summary:"<<std::endl;
  std::cout<<"#Size  n    |  Avg. Time (s) |   Min. Time(s) |   Max. Time(s) | σ Time(s)"<<std::endl;
  std::cout<<n<<" "<<timeAvg<<" "<<tmin<<" "<<tmax<<" "<<sig<<std::endl;

  cudaFree (devPtrA);
  cudaFree (devPtrB);
  cublasDestroy(handle);
  delete[] times;
  delete[] A;
  delete[] B;

  return 0;
}

