#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t err__ = (call);                                             \
    if (err__ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
                   static_cast<int>(err__));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace {

constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int BM_ASYNC = 128;
constexpr int BN_ASYNC = 128;
constexpr int BK_ASYNC = 64;
constexpr int WARPS_PER_BLOCK = 16;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WARPS_PER_BLOCK_ASYNC = 8;
constexpr int THREADS_PER_BLOCK_ASYNC = WARPS_PER_BLOCK_ASYNC * 32;

// Padding keeps the shared-memory leading dimensions WMMA-friendly while
// reducing bank conflicts during ldmatrix-style loads.
constexpr int SKEW_HALF = 8;
constexpr int AS_LD = BM + SKEW_HALF;
constexpr int BS_LD = BK + SKEW_HALF;
constexpr int AS_LD_ASYNC = BM_ASYNC + SKEW_HALF;
constexpr int BS_LD_ASYNC = BK_ASYNC + SKEW_HALF;
constexpr int ASYNC_STAGES = 2;
constexpr std::size_t ASYNC_SMEM_BYTES =
    ASYNC_STAGES * (BK_ASYNC * AS_LD_ASYNC + BN_ASYNC * BS_LD_ASYNC) *
    sizeof(half);

__device__ __forceinline__ half to_half(float x) { return __float2half_rn(x); }
__device__ __forceinline__ half to_half(half x) { return x; }

__device__ __forceinline__ void cp_async_16(void *smem_ptr,
                                            const void *gmem_ptr) {
  unsigned smem_addr =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

template <typename InT>
__global__ void wmma_gemm_256x128_bcol_kernel(int dim_m, int dim_n, int dim_k,
                                              const InT *__restrict__ d_a,
                                              const InT *__restrict__ d_b,
                                              float *__restrict__ d_c) {
  const int block_m = BM * blockIdx.x;
  const int block_n = BN * blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int warp_m = warp_id >> 1; // 0, 0, 1, 1, ..., 7, 7
  const int warp_n = warp_id & 1;  // 0, 1, 0, 1, ..., 0, 1

  __shared__ half block_a[BK][AS_LD];
  __shared__ half block_b[BN][BS_LD];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  for (int k0 = 0; k0 < dim_k; k0 += BK) {
    // A is column-major: A[m + k * dim_m].  This mapping makes each half-warp
    // load a contiguous run of M values for the same K.
    for (int idx = tid; idx < BK * BM; idx += THREADS_PER_BLOCK) {
      const int local_m = idx % BM;
      const int local_k = idx / BM;
      block_a[local_k][local_m] =
          to_half(d_a[(k0 + local_k) * dim_m + block_m + local_m]);
    }

    // Keep B in shared memory as column-major KxN: block_b[n][k].  This
    // matches the global memory layout, so adjacent lanes load and store
    // adjacent K values without an explicit transpose into shared memory.
    for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
      const int local_k = idx % BK;
      const int local_n = idx / BK;
      block_b[local_n][local_k] =
          to_half(d_b[(block_n + local_n) * dim_k + k0 + local_k]);
    }

    __syncthreads();

    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int a_tile = warp_m * 2 + r;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>
            a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[kk][a_tile * 16], AS_LD);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int b_tile = warp_n * 4 + c;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
              b_frag;
          wmma::load_matrix_sync(b_frag, &block_b[b_tile * 16][kk], BS_LD);
          wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int c_m = block_m + (warp_m * 2 + r) * 16;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int c_n = block_n + (warp_n * 4 + c) * 16;
      wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m,
                              wmma::mem_col_major);
    }
  }
}

__device__ __forceinline__ void
load_half_stage_async(int stage, int k0, int block_m, int block_n, int dim_m,
                      int dim_k, const half *__restrict__ d_a,
                      const half *__restrict__ d_b,
                      half (*block_a)[BK_ASYNC][AS_LD_ASYNC],
                      half (*block_b)[BN_ASYNC][BS_LD_ASYNC]) {
  const int tid = threadIdx.x;

  for (int vec = tid; vec < BK_ASYNC * (BM_ASYNC / 8);
       vec += THREADS_PER_BLOCK_ASYNC) {
    const int local_k = vec / (BM_ASYNC / 8);
    const int local_m = (vec % (BM_ASYNC / 8)) * 8;
    cp_async_16(&block_a[stage][local_k][local_m],
                &d_a[(k0 + local_k) * dim_m + block_m + local_m]);
  }

  for (int vec = tid; vec < BN_ASYNC * (BK_ASYNC / 8);
       vec += THREADS_PER_BLOCK_ASYNC) {
    const int local_n = vec / (BK_ASYNC / 8);
    const int local_k = (vec % (BK_ASYNC / 8)) * 8;
    cp_async_16(&block_b[stage][local_n][local_k],
                &d_b[(block_n + local_n) * dim_k + k0 + local_k]);
  }
}

__global__ void wmma_gemm_128x128_bcol_async_half_kernel(
    int dim_m, int dim_n, int dim_k, const half *__restrict__ d_a,
    const half *__restrict__ d_b, float *__restrict__ d_c) {
  const int block_m = BM_ASYNC * blockIdx.x;
  const int block_n = BN_ASYNC * blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int warp_m = warp_id >> 1;
  const int warp_n = warp_id & 1;

  extern __shared__ half smem[];
  half (*block_a)[BK_ASYNC][AS_LD_ASYNC] =
      reinterpret_cast<half (*)[BK_ASYNC][AS_LD_ASYNC]>(smem);
  half (*block_b)[BN_ASYNC][BS_LD_ASYNC] =
      reinterpret_cast<half (*)[BN_ASYNC][BS_LD_ASYNC]>(
          smem + ASYNC_STAGES * BK_ASYNC * AS_LD_ASYNC);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  load_half_stage_async(0, 0, block_m, block_n, dim_m, dim_k, d_a, d_b,
                        block_a, block_b);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  int stage = 0;
  for (int k0 = 0; k0 < dim_k; k0 += BK_ASYNC) {
    const int next_stage = stage ^ 1;
    const int next_k = k0 + BK_ASYNC;
    if (next_k < dim_k) {
      load_half_stage_async(next_stage, next_k, block_m, block_n, dim_m,
                            dim_k, d_a, d_b, block_a, block_b);
      cp_async_commit();
    }

    for (int kk = 0; kk < BK_ASYNC; kk += 16) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int a_tile = warp_m * 2 + r;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>
            a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[stage][kk][a_tile * 16],
                               AS_LD_ASYNC);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int b_tile = warp_n * 4 + c;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
              b_frag;
          wmma::load_matrix_sync(b_frag, &block_b[stage][b_tile * 16][kk],
                                 BS_LD_ASYNC);
          wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
        }
      }
    }

    if (next_k < dim_k) {
      cp_async_wait_all();
    }
    __syncthreads();
    stage = next_stage;
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int c_m = block_m + (warp_m * 2 + r) * 16;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int c_n = block_n + (warp_n * 4 + c) * 16;
      wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m,
                              wmma::mem_col_major);
    }
  }
}

__global__ void convert_to_half_kernel(const float *__restrict__ in,
                                       half *__restrict__ out,
                                       std::size_t count) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  for (std::size_t i = idx; i < count; i += stride) {
    out[i] = __float2half_rn(in[i]);
  }
}

template <typename F>
float time_ms(F &&fn, int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / repeat;
}

struct ErrorStats {
  double mean_abs;
  double max_abs;
};

ErrorStats compare_on_host(const float *d_ref, const float *d_test,
                           std::size_t count) {
  std::vector<float> h_ref(count);
  std::vector<float> h_test(count);
  CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  long double sum = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double e = std::abs(static_cast<double>(h_ref[i]) -
                              static_cast<double>(h_test[i]));
    sum += e;
    max_abs = std::max(max_abs, e);
  }
  return {static_cast<double>(sum / count), max_abs};
}

void print_result(const char *name, float ms, int m, int n, int k,
                  ErrorStats err) {
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  const double tflops = flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
  std::printf("%-28s %8.3f ms  %8.2f TFLOP/s  mean_abs=%g  max_abs=%g\n",
              name, ms, tflops, err.mean_abs, err.max_abs);
}

void print_result_no_error(const char *name, float ms, int m, int n, int k) {
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  const double tflops = flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
  std::printf("%-28s %8.3f ms  %8.2f TFLOP/s\n", name, ms, tflops);
}

} // namespace

int main(int argc, char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  int repeat = 10;
  if (argc > 1)
    m = std::atoi(argv[1]);
  if (argc > 2)
    k = std::atoi(argv[2]);
  if (argc > 3)
    n = std::atoi(argv[3]);
  if (argc > 4)
    repeat = std::atoi(argv[4]);

  if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0) {
    std::fprintf(stderr, "Usage: %s [m k n repeat]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if ((m % BM) != 0 || (n % BN) != 0 || (k % BK_ASYNC) != 0) {
    std::fprintf(stderr,
                 "This fast kernel expects m %% %d == 0, n %% %d == 0, "
                 "and k %% %d == 0.\n",
                 BM, BN, BK_ASYNC);
    return EXIT_FAILURE;
  }

  const std::size_t a_count = static_cast<std::size_t>(m) * k;
  const std::size_t b_count = static_cast<std::size_t>(k) * n;
  const std::size_t c_count = static_cast<std::size_t>(m) * n;
  std::printf("Matrix sizes: M=%d K=%d N=%d, repeat=%d\n", m, k, n, repeat);

  std::vector<float> h_a(a_count);
  std::vector<float> h_b(b_count);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < a_count; ++i)
    h_a[i] = dist(rng);
  for (std::size_t i = 0; i < b_count; ++i)
    h_b[i] = dist(rng);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c_ref = nullptr;
  float *d_c_test = nullptr;
  half *d_a_half = nullptr;
  half *d_b_half = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c_ref, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c_test, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_half, a_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_b_half, b_count * sizeof(half)));
  CUDA_CHECK(
      cudaMemcpy(d_a, h_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b, h_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

  const int convert_threads = 256;
  const int convert_blocks = 4096;
  convert_to_half_kernel<<<convert_blocks, convert_threads>>>(d_a, d_a_half,
                                                              a_count);
  convert_to_half_kernel<<<convert_blocks, convert_threads>>>(d_b, d_b_half,
                                                              b_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(m / BM, n / BN);
  dim3 block_async(THREADS_PER_BLOCK_ASYNC);
  dim3 grid_async(m / BM_ASYNC, n / BN_ASYNC);
  const int warmup = 2;
  CUDA_CHECK(cudaFuncSetAttribute(
      wmma_gemm_128x128_bcol_async_half_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(ASYNC_SMEM_BYTES)));

  auto cublas_f32_ms = time_ms(
      [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a,
            CUDA_R_32F, m, d_b, CUDA_R_32F, k, &beta, d_c_ref, CUDA_R_32F, m,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      },
      warmup, repeat);
  print_result_no_error("cuBLAS f32 FAST_16F", cublas_f32_ms, m, n, k);

  auto custom_f32_ms = time_ms(
      [&]() {
        wmma_gemm_256x128_bcol_kernel<float>
            <<<grid, block>>>(m, n, k, d_a, d_b, d_c_test);
        CUDA_CHECK(cudaGetLastError());
      },
      warmup, repeat);
  CUDA_CHECK(cudaDeviceSynchronize());
  ErrorStats f32_err = compare_on_host(d_c_ref, d_c_test, c_count);
  print_result("custom WMMA f32 inputs", custom_f32_ms, m, n, k, f32_err);

  auto cublas_half_ms = time_ms(
      [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a_half,
            CUDA_R_16F, m, d_b_half, CUDA_R_16F, k, &beta, d_c_ref, CUDA_R_32F,
            m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      },
      warmup, repeat);
  print_result_no_error("cuBLAS half inputs", cublas_half_ms, m, n, k);

  auto custom_half_ms = time_ms(
      [&]() {
        wmma_gemm_256x128_bcol_kernel<half>
            <<<grid, block>>>(m, n, k, d_a_half, d_b_half, d_c_test);
        CUDA_CHECK(cudaGetLastError());
      },
      warmup, repeat);
  CUDA_CHECK(cudaDeviceSynchronize());
  ErrorStats half_err = compare_on_host(d_c_ref, d_c_test, c_count);
  print_result("custom WMMA half inputs", custom_half_ms, m, n, k, half_err);

  auto custom_half_async_ms = time_ms(
      [&]() {
        wmma_gemm_128x128_bcol_async_half_kernel<<<grid_async, block_async,
                                                    ASYNC_SMEM_BYTES>>>(
            m, n, k, d_a_half, d_b_half, d_c_test);
        CUDA_CHECK(cudaGetLastError());
      },
      warmup, repeat);
  CUDA_CHECK(cudaDeviceSynchronize());
  ErrorStats half_async_err = compare_on_host(d_c_ref, d_c_test, c_count);
  print_result("custom WMMA half async", custom_half_async_ms, m, n, k,
               half_async_err);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c_ref));
  CUDA_CHECK(cudaFree(d_c_test));
  CUDA_CHECK(cudaFree(d_a_half));
  CUDA_CHECK(cudaFree(d_b_half));
  return 0;
}
