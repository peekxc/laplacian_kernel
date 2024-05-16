#include "laplacian.h"

// Kernel for adding the boundary/coboundary of r 
__global__ void add_boundaries(const index_t k, const index_t n, const index_t N, const index_t** BT, const float* x, float *y)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N){
    index_t ps[4] = { 0, 0, 0, 0 }; 
    k_boundary(n, tid, k - 1, BT, (index_t*) ps);
    index_t i = ps[0], j = ps[1], q = ps[2];
    atomicAdd(y + i, -x[j]);
    atomicAdd(y + j, -x[j]);
    atomicAdd(y + i, x[q]);
    atomicAdd(y + q, x[i]);
    atomicAdd(y + j, -x[q]);
    atomicAdd(y + q, -x[j]);
  }  
}

void compute_deg_full(const int n, const int k, const index_t** BT, index_t* deg){
  for (size_t r = 0; r < binom(n, k); ++r){
    index_t ps[4] = { 0, 0, 0, 0 }; 
    k_boundary(n, r, k - 1, BT, (const index_t*) ps);
    deg[ps[0]] += 1;
    deg[ps[1]] += 1;
    deg[ps[2]] += 1;
  }
}