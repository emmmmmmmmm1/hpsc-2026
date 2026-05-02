#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 vec_xj = _mm512_load_ps(x);
  __m512 vec_yj = _mm512_load_ps(y);
  __m512 vec_mj = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 vec_xi = _mm512_set1_ps(x[i]);
    __m512 vec_yi = _mm512_set1_ps(y[i]);

    __m512 vec_rx = _mm512_sub_ps(vec_xi, vec_xj);
    __m512 vec_ry = _mm512_sub_ps(vec_yi, vec_yj);

    __m512 vec_rx2 = _mm512_mul_ps(vec_rx, vec_rx);
    __m512 vec_ry2 = _mm512_mul_ps(vec_ry, vec_ry);
    __m512 vec_r2 = _mm512_add_ps(vec_rx2, vec_ry2);

    __m512 vec_inv_r = _mm512_rsqrt14_ps(vec_r2);

    __m512 vec_inv_r3 = _mm512_mul_ps(vec_inv_r, _mm512_mul_ps(vec_inv_r, vec_inv_r));

    __m512 vec_m_inv_r3 = _mm512_mul_ps(vec_mj, vec_inv_r3);

    __m512 vec_fx = _mm512_mul_ps(vec_rx, vec_m_inv_r3);
    __m512 vec_fy = _mm512_mul_ps(vec_ry, vec_m_inv_r3);

    __mmask16 mask = ~(1 << i);

    float sum_fx = _mm512_mask_reduce_add_ps(mask, vec_fx);
    float sum_fy = _mm512_mask_reduce_add_ps(mask, vec_fy);

    fx[i] -= sum_fx;
    fy[i] -= sum_fy;

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
