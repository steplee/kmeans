#include <iostream>
#include <vector>
#include <algorithm>

#include "helper_cuda.h"

#include <cstdlib>
#include <cstdio>

using namespace std;

const int K = K_FLAG; // 224
const int N = 32 * 1565; // 50080

struct point { float y; float x; };

point operator+(const point& a, const point& b) {
  return {a.y+b.y, a.x+b.x};
}
__host__ __device__ void operator+=(point& a, const point& b) {
  a.x += b.x;
  a.y += b.y;
}
void operator-=(point& a, const point& b) {
  a.x -= b.x;
  a.y -= b.y;
}
__device__ __host__ void operator/=(point& a, float f) {
  a.x /= f;
  a.y /= f;
}
__device__ __host__ point operator/(point& a, float f) {
  return {a.y/f, a.x/f};
}
ostream& operator<<(ostream& o, const point& a) {
  return o << "{" << a.y << ", " << a.x << "} ";
}

static float frand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

__host__ __device__ float dist(const point& a, const point& b) {
  //return sqrt( (a.y-b.y)*(a.y-b.y) + (a.y-b.y)(a.y-b.y) );
  return  (a.y-b.y)*(a.y-b.y) + (a.x-b.x)*(a.x-b.x);
}

__global__ void assign_pt(const point* pts, const point* mus, int* membership) {
  //int id = blockIdx.x*blockDim.x + threadIdx.x;
  int id = threadIdx.x;

  float best = 999999;

  for (int m=0; m<K; m++) {
    float d = dist(pts[id], mus[m]);
    if (d < best) {
      membership[id] = m;
      best = d;
    }
  }

  __threadfence_system();
}

__global__ void set_mu(point* mus, const point* pts, const int* membership) {
  //int id = blockIdx.x*blockDim.x + threadIdx.x;
  int id = threadIdx.x;

  // Having a local buffer may reduce writes to shared/global memory ??
  point local{0,0};
  int cnt = 0;

  for (int p=0; p<N; p++) {
    if (membership[p] == id) {
      local += pts[p];
      cnt++;
    }
  }

  mus[id] = local;

  if ( cnt > 0 )
    mus[id] /= ((float)cnt);

  __threadfence_system();
}


int main(int argc, char** argv) {

  point *pts, *mus;
  int *membership;

  pts = (point*) malloc( N * sizeof(point) );
  mus = (point*) malloc( K * sizeof(point) );
  membership = (int*) malloc( N * sizeof(int) );

  // Cuda setup.
  int dev = findCudaDevice(argc, (const char **)argv);

  srand(0);

  // Data.
  for (int i=0; i<N; i++) 
    pts[i] = {frand(), frand()},
    membership[i] = rand() % K;
  for (int i=0; i<K; i++) 
    mus[i] = {frand(), frand()};

  // TODO see if counting helps even with cuda.
  // I suspect it won't since we have to write to global memory sooo much.
  // Maybe write to shared & batch update?
  int mu_cnts[K];
  memset(mu_cnts, 0, sizeof(int)*K);

  point* d_mus;
  point* d_pts;
  int* d_membership;

  checkCudaErrors(cudaMalloc((void**)&d_mus, sizeof(point)*K));
  checkCudaErrors(cudaMalloc((void**)&d_pts, sizeof(point)*N));
  checkCudaErrors(cudaMalloc((void**)&d_membership, sizeof(int)*N));
  checkCudaErrors(cudaMemcpy(d_mus, mus, sizeof(point)*K, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pts, pts, sizeof(point)*N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_membership, membership, sizeof(int)*N, cudaMemcpyHostToDevice));

  // Run.
  for (int i=0; i<100; i++) {
    //cout << " (iter " << i << ")\n";

    // Phase 1: update point memberships
    assign_pt<<< 1, N >>>(d_pts, d_mus, d_membership);

    // Phase 2: update means
    set_mu<<< 1, K >>>(d_mus, d_pts, d_membership);

    //cudaEventSynchronize(cudaEventBlockingSync);
  }



  checkCudaErrors(cudaMemcpy(mus, d_mus, sizeof(point)*K, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(pts, d_pts, sizeof(point)*N, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(membership, d_membership, sizeof(int)*N, cudaMemcpyDeviceToHost));

  // Eval.
  for (int m=210; m<K; m++) {
    int cnt = 0;
    for (int p=0; p<N; p++)
      if (membership[p] == m)
          cnt += 1;
    cout << " mu" << m << " has " << cnt << " points at " << mus[m] << ".\n";
  }

  return 0;
}
