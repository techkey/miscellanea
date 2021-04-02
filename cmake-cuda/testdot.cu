#include <iostream>
#include <chrono>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CUDABLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif
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
  double *pi = new double[nSamples];
  double *diff = new double[nSamples];
  int N = n+nSamples;
  double *A = new double[N];
  double *B = new double[N];
  double tmin=std::numeric_limits<double>::max();
  double tmax=0.0;
  double z;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  double* devPtrA;
  double* devPtrB;
  
  // set arrays values
  int sign = 1;
  for (int j=0; j<N; ++j) {
    A[j] = 4./(double)(2*j+1);
    B[j] = (double)(sign);
    sign *= -1;
  }
#ifdef CUDABLAS
  cudaStat = cudaMalloc ((void**)&devPtrA, (N)*sizeof(*A));
  if (cudaStat != cudaSuccess) {
    std::cout << "device memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc ((void**)&devPtrB, (N)*sizeof(*B));
  if (cudaStat != cudaSuccess) {
    std::cout << "device memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS initialization failed" << std::endl;
      return EXIT_FAILURE;
  }
#endif
  for (int i=0; i<nSamples; ++i) {

#ifdef CUDABLAS
    stat = cublasSetVector(n+i, sizeof(double),
                A, 1, devPtrA , 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << "CUBLAS SetVector failed" << std::endl;
        return EXIT_FAILURE;
    }
    stat = cublasSetVector(n+i, sizeof(double),
                B, 1, devPtrB , 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << "CUBLAS SetVector failed" << std::endl;
        return EXIT_FAILURE;
    }
#endif

    const auto start = std::chrono::high_resolution_clock::now();
#ifdef WITH_BLAS
    z=cblas_ddot(n+i,A,1,B,1);
#else
#ifdef CUDABLAS
    stat = cublasDdot(handle, n+i, devPtrA, 1, devPtrB, 1, &z);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << "CUBLAS Ddot failed" << std::endl;
        return EXIT_FAILURE;
    }
#endif
//    z=dot(A,B,n);
#endif    
    pi[i] = z;
    diff[i] = M_PI - z;
    times[i] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    if (tmin>times[i]) tmin=times[i];
    if (tmax<times[i]) tmax=times[i];

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
    // std::cout<<i+1<<" "<<n<<" "<<times[i]<<std::endl;
    std::cout<<i<<" "<<pi[i]<<" |z - pi| = "<< fabs(diff[i]) <<" < "<< (4./(double)(2*(i+n)+3)) << " " << times[i]<<"s "<<std::endl;
  }
  std::cout<<"Last inner product: "<<z<<std::endl;
  std::cout<<"Summary:"<<std::endl;
  std::cout<<"#Size  n    |  Avg. Time (s) |   Min. Time(s) |   Max. Time(s) | σ Time(s)"<<std::endl;
  std::cout<<n<<" "<<timeAvg<<" "<<tmin<<" "<<tmax<<" "<<sig<<std::endl;
  
#ifdef CUDABLAS
  cudaFree (devPtrA);
  cudaFree (devPtrB);
  cublasDestroy(handle);
#endif

  delete[] times;
  delete[] A;
  delete[] B;
  std::cout<<"Last inner product: "<<z<<std::endl;
  return 0;
  }

