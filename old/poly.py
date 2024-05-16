import numpy as np 
import cupy as cp
import numba as nb 

from numba import float32, float64, int32, int64, cuda
# template< std::floating_point F >
# inline void poly(F x, const F mu_sqrt_rec, const F* a, const F* b, F* z, const size_t n) noexcept {
#   // assert(z.size() == a.size());
#   z[0] = mu_sqrt_rec;
#   z[1] = (x - a[0]) * z[0] / b[1];
#   for (size_t i = 2; i < n; ++i) {
#     // F zi = ((x - a[i-1]) * z[i-1] - b[i-1] * z[i-2]) / b[i];
#     // Slightly more numerically stable way of doing the above
#     F s = (x - a[i-1]) / b[i];
#     F t = -b[i-1] / b[i];
#     z[i] = s * z[i-1] + t * z[i-2];
#   }
# }

def poly(x: np.ndarray, mu_sqrt_rec: float64, a: np.ndarray, b: np.ndarray, z: np.ndarray, n: int64):
  z[0] = mu_sqrt_rec
  z[1] = (x - a[0]) * z[0] / b[1]
  for i in range(2, n):
    s = (x - a[i-1]) / b[i]
    t = -b[i-1] / b[i]
    z[i] = s * z[i-1] + t * z[i-2]
  
# template< std::floating_point F >
# void FTTR_weights(const F* theta, const F* alpha, const F* beta, const size_t k, F* weights) {
#   // assert(ew.size() == a.size());
#   const auto a = Eigen::Map< const Array< F > >(alpha, k);
#   const auto b = Eigen::Map< const Array< F > >(beta, k);
#   const auto ew = Eigen::Map< const Array< F > >(theta, k); 
#   const F mu_0 = ew.abs().sum();
#   const F mu_sqrt_rec = 1.0 / std::sqrt(mu_0);
#   Array< F > p(a.size());
#   for (size_t i = 0; i < k; ++i){
#     poly(theta[i], mu_sqrt_rec, a.data(), b.data(), p.data(), a.size());
#     F weight = 1.0 / p.square().sum();
#     weights[i] = weight / mu_0; 
#   }
# }
def FTTR_weights(theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray, k: int64, weights: np.ndarray):
  pass
