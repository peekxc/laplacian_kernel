#include <cinttypes> 
using index_t = int64_t;

// Recursive binomial coefficient calculation - supports degenerate cases 
// Baseline from: https://stackoverflow.com/questions/44718971/calculate-binomial-coffeficient-very-reliably
// Requires O(min{k,n-k}), uses pascals triangle approach (+ degenerate cases)
__device__ size_t binom(size_t n, size_t k) {
  return
    (k > n) ? 0 :                  // out of range
    (k == 0 || k == n) ? 1 :       // edge
    (k == 1 || k == n-1) ? n :     // first
    (k+k < n) ?                    // recursive:
    (binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
    (binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
}

// Precomputes a table of binomial coefficients
void precompute_BC(const index_t n, const index_t k, index_t** BT){
  for (index_t i = 0; i <= n; ++i) {
    BT[0][i] = 1;
    for (index_t j = 1; j < std::min(i, k + 1); ++j){
      BT[j][i] = binom(i,j); // BT[j - 1][i - 1] + BT[j][i - 1];
    }
    if (i <= k) { BT[i][i] = 1; };
  }
}

__device__ index_t get_max(index_t top, index_t bottom, const index_t r, const index_t m, const index_t** BT) noexcept {
  if (!(BT[m][bottom] <= r)) { return bottom; }
  index_t size = (top - bottom);
  while (size > 0){
    index_t step = size >> 1;
    index_t mid = top - step;
    if (BT[m][mid] <= r){
      top = mid - 1;
      size -= step + 1;
    } else {
      size = step;
    }
  }
  return top;
}

__device__ index_t get_max_vertex(const index_t r, const index_t m, const index_t n, const index_t** BT) noexcept {
  index_t k_lb = m - 1;
  return 1 + get_max(n, k_lb, r, m, BT);
}

__device__ void k_boundary(const index_t n, const index_t simplex, const index_t dim, const index_t** BT, index_t* br) {
  index_t idx_below = simplex;
  index_t idx_above = 0; 
  index_t j = n - 1;
  bool cont_enum = true; 
  for (index_t k = dim; k >= 0 && cont_enum; --k){
    j = get_max_vertex(idx_below, k + 1, j, BT) - 1; // NOTE: Danger!
    index_t c = BT[k+1][j];
    index_t face_index = idx_above - c + idx_below;
    idx_below -= c;
    idx_above += BT[k][j];
    br[dim-k] = face_index;
  }
}

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