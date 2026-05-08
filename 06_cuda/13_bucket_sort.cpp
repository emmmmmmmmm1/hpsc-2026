#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void countBuckets(int *d_key, int *d_bucket, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(&d_bucket[d_key[tid]], 1);
    }
}

__global__ void writeSorted(int *d_key, int *d_offsets, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < range) {
        int start_idx = d_offsets[i];
        int end_idx = d_offsets[i + 1];
        for (int j = start_idx; j < end_idx; j++) {
            d_key[j] = i;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;
    std::vector<int> h_key(n);

    for (int i = 0; i < n; i++) {
        h_key[i] = rand() % range;
        printf("%d ", h_key[i]);
    }
    printf("\n");

    int *d_key, *d_bucket, *d_offsets;
    cudaMalloc((void**)&d_key, n * sizeof(int));
    cudaMalloc((void**)&d_bucket, range * sizeof(int));
    cudaMalloc((void**)&d_offsets, (range + 1) * sizeof(int));

    cudaMemcpy(d_key, h_key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_bucket, 0, range * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    countBuckets<<<blocksPerGrid, threadsPerBlock>>>(d_key, d_bucket, n);
    
    cudaDeviceSynchronize();

    std::vector<int> h_bucket(range);
    cudaMemcpy(h_bucket.data(), d_bucket, range * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_offsets(range + 1, 0);
    for (int i = 0; i < range; i++) {
        h_offsets[i + 1] = h_offsets[i] + h_bucket[i];
    }

    cudaMemcpy(d_offsets, h_offsets.data(), (range + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int writeBlocks = (range + threadsPerBlock - 1) / threadsPerBlock;
    writeSorted<<<writeBlocks, threadsPerBlock>>>(d_key, d_offsets, range);
    cudaDeviceSynchronize();

    cudaMemcpy(h_key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d ", h_key[i]);
    }
    printf("\n");

    cudaFree(d_key);
    cudaFree(d_bucket);
    cudaFree(d_offsets);

    return 0;
}
