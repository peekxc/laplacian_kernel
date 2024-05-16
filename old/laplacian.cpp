#include "laplacian.h"

// Up-laplacian matvec for dimension (1/2)-simplices 
void laplacian1_matvec(const index_t k, const index_t n, const index_t N, const index_t** BT, const float* x, float *y){
  for (size_t tid = 0; tid < N; ++tid){
    index_t ps[4] = { 0, 0, 0, 0 }; 
    k_boundary(n, tid, k - 1, BT, (index_t*) ps);
    index_t i = ps[0], j = ps[1], q = ps[2];
    y[i] -= x[j];
    y[j] -= x[j];
    y[i] += x[q];
    y[q] += x[i];
    y[j] -= x[q];
    y[q] -= x[j];
  }
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(_claplacian, m) {
  m.doc() = "pybind11 example module";
  m.def("laplacian1_matvec", [](
    const index_t k, const index_t n, const index_t N,
    const py::array_t< float >& _x, py::array_t< float >& _y
  ) {
    // const index_t** BT = _BT.data()
    index_t** BT = new index_t*[k+2];
    for (auto i = 0; i < (k+2); ++i){
      BT[i] = new index_t[n];
      for (auto j = 0; j < n; ++j){
        BT[i][j] = binom(j,i);
      }
    }
    
    const float* x = _x.data();   
    float* y = _y.mutable_data();
    laplacian1_matvec(k, n, N, (const index_t**) BT, x, y);
  });
}