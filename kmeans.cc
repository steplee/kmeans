#include <iostream>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>

using namespace std;

const int K = K_FLAG; // 224
const int N = 32 * 1565; // 50080

struct point { float y; float x; };

inline point operator+(const point& a, const point& b) {
  return {a.y+b.y, a.x+b.x};
}
inline void operator+=(point& a, const point& b) {
  a.x += b.x;
  a.y += b.y;
}
inline void operator-=(point& a, const point& b) {
  a.x -= b.x;
  a.y -= b.y;
}
inline void operator/=(point& a, float f) {
  a.x /= f;
  a.y /= f;
}
inline point operator/(point& a, float f) {
  return {a.y/f, a.x/f};
}
inline ostream& operator<<(ostream& o, const point& a) {
  return o << "{" << a.y << ", " << a.x << "} ";
}

inline static float frand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
inline static float dist(point& a, point& b) {
  //return sqrt( (a.y-b.y)*(a.y-b.y) + (a.y-b.y)(a.y-b.y) );
  return  (a.y-b.y)*(a.y-b.y) + (a.x-b.x)*(a.x-b.x);
}



int main(int argc, char** argv) {

  point *pts, *mus;
  int *membership;

  pts = (point*) malloc( N * sizeof(point) );
  mus = (point*) malloc( K * sizeof(point) );
  membership = (int*) malloc( N * sizeof(int) );

  srand(0);

  // Data.
  for (int i=0; i<N; i++) 
    pts[i] = {frand(), frand()},
    membership[i] = rand() % K;
  for (int i=0; i<K; i++) 
    mus[i] = {frand(),frand()};

  int mu_cnts[K];
  point mu_centers[K];
  memset(mu_cnts, 0, sizeof(int)*K);
  memset(mu_centers, 0, sizeof(point)*K);

  // Run.
  for (int i=0; i<100; i++) {
    //cout << " (iter " << i << ")\n";
    memset(mu_cnts, 0, sizeof(int)*K);
    memset(mu_centers, 0, sizeof(point)*K);

    // Phase 1: update point memberships
    for (int p=0; p<N; p++) {
      float best = 999999;
      membership[p] = K+1;

      for (int m=0; m<K; m++) {
        float d = dist(pts[p], mus[m]);
        if (d < best) {
          if(membership[p] < K)
            mu_centers[membership[p]] -= pts[p],
            mu_cnts[membership[p]]--;

          mu_centers[m] += pts[p];
          mu_cnts[m]++;

          membership[p] = m;
          best = d;
        }
      }
    }

    // Phase 2: update means
    for (int m=0; m<K; m++) {
      if (mu_cnts[m] > 0)
        mus[m] = mu_centers[m] / ((float)mu_cnts[m]);
    }
  }

  // Eval.
  if (argc > 1 and strcmp(argv[1], "show") == 0)
    for (int m=0; m<K; m++) {
      int cnt = 0;
      for (int p=0; p<N; p++)
        if (membership[p] == m)
            cnt += 1;
      cout << " mu" << m << " has " << cnt << " points at " << mus[m] << ".\n";
    }

  return 0;
}
