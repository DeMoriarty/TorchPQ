#define _VOLATILE_ 
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

typedef unsigned char uint8_t;
typedef long long ll_t;

typedef struct __builtin_align__(8)
{
  float value;
  float index;
} pair;

typedef struct __device_builtin__ __builtin_align__(_NCS_)
{
  uint8_t _VARNAMES_;
} _uint8n_t;

typedef union {
  _uint8n_t u8n;
  uint8_t val[_NCS_];
} uint8n_t;

__device__ __forceinline__ float atomicMax(float *address, float val)
{
  int ret = __float_as_int(*address);
  while(val > __int_as_float(ret))
  {
    int old = ret;
    if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        break;
  }
  return __int_as_float(ret);
}

__device__ __forceinline__ unsigned int bfe(
  unsigned int source,
  unsigned int bitIndex
) {
  unsigned int bit;
  asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bit) : "r"((unsigned int) source), "r"(bitIndex), "r"(1));
  return bit;
}

__device__ __forceinline__ void warp_comparator(
  float &value,
  float &index,
  const int stride,
  const int direction
){
  const float otherValue = __shfl_xor_sync(0xFFFFFFFF, value, stride);
  const float otherIndex = __shfl_xor_sync(0xFFFFFFFF, index, stride);
  bool condition = value < otherValue == direction;
  index = condition ? otherIndex : index;
  value = condition ? otherValue : value;
}

__device__ __forceinline__ void block_comparator(
  float &value,
  float &index,
  const int stride,
  const int direction,
  const int laneID,
  _VOLATILE_ float sMem[]
){
  float tempPrecomputed1 = sMem[laneID];
  float tempPrecomputed2 = sMem[_TPB_ + laneID];
  __syncthreads();

  sMem[laneID] = value;
  sMem[_TPB_ + laneID] = index;
  __syncthreads();

  float otherValue = sMem[laneID ^ stride];
  float otherIndex = sMem[_TPB_ + laneID ^ stride];
  __syncthreads();

  sMem[laneID] = tempPrecomputed1;
  sMem[_TPB_ + laneID] = tempPrecomputed2;
  __syncthreads();

  bool condition = value < otherValue == direction;
  value = condition ? otherValue : value;
  index = condition ? otherIndex : index;
  /*
  */
}

__device__ __forceinline__ void block_comparator_noop(
){
  __syncthreads();
  __syncthreads();
  __syncthreads();
  __syncthreads();
}

__device__ __forceinline__ void thread_comparator(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  const int direction
){
  bool condition = value > otherValue == direction;
  if (condition){
    value = otherValue;
    index = otherIndex;
  }
}

__device__ void bitonic_sort_2(
  float &value,
  float &index,
  int laneID
){
  warp_comparator(value, index, 1, bfe(laneID, 1) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_4(
  float &value,
  float &index,
  int laneID
){
  bitonic_sort_2(value, index, laneID);
  warp_comparator(value, index, 2, bfe(laneID, 2) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 2) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_8(
  float &value,
  float &index,
  int laneID
){
  bitonic_sort_4(value, index, laneID);
  warp_comparator(value, index, 4, bfe(laneID, 3) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 3) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 3) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_16(
  float &value,
  float &index,
  int laneID
){
  bitonic_sort_8(value, index, laneID);
  warp_comparator(value, index, 8, bfe(laneID, 4) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 4) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 4) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 4) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_32(
  float &value,
  float &index,
  int laneID
){
  bitonic_sort_16(value, index, laneID);
  warp_comparator(value, index, 16, bfe(laneID, 5) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 5) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 5) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 5) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 5) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_global_2(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  int laneID
) {
  if (_TPB_ - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

__device__ void bitonic_sort_global_4(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  int laneID
) {
  if (_TPB_ - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

__device__ void bitonic_sort_global_8(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  int laneID
) {
  if (_TPB_ - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

__device__ void bitonic_sort_global_16(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  int laneID
) {
  if (_TPB_ - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

__device__ void bitonic_sort_global_32(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  int laneID
) {
  if (_TPB_ - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

#if _TPB_ >= 64
__device__ void bitonic_sort_64(
  float &value,
  float &index,
  _VOLATILE_ float sMem[],
  int laneID
){
  bitonic_sort_32(value, index, laneID);
  block_comparator(value, index, 32, bfe(laneID, 6) ^ bfe(laneID, 5), laneID, sMem);
  warp_comparator(value, index, 16, bfe(laneID, 6) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 6) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 6) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 6) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 6) ^ bfe(laneID, 0));
}
#endif
__device__ void bitonic_sort_global_64(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  _VOLATILE_ float sMem[],
  int laneID
) {
  if (_TPB_ - 64 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, sMem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));

    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
  }
}

#if _TPB_ >= 128
__device__ void bitonic_sort_128(
  float &value,
  float &index,
  _VOLATILE_ float sMem[],
  int laneID
){
  bitonic_sort_64(value, index, sMem, laneID);
  block_comparator(value, index, 64, bfe(laneID, 7) ^ bfe(laneID, 6), laneID, sMem);
  block_comparator(value, index, 32, bfe(laneID, 7) ^ bfe(laneID, 5), laneID, sMem);
  warp_comparator(value, index, 16, bfe(laneID, 7) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 7) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 7) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 7) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 7) ^ bfe(laneID, 0));
}
#endif
__device__ void bitonic_sort_global_128(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  _VOLATILE_ float sMem[],
  int laneID
) {
  if (_TPB_ - 128 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, sMem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, sMem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
  }
}

#if _TPB_ >= 256
__device__ void bitonic_sort_256(
  float &value,
  float &index,
  _VOLATILE_ float sMem[],
  int laneID
){
  bitonic_sort_128(value, index, sMem, laneID);
  block_comparator(value, index, 128, bfe(laneID, 8) ^ bfe(laneID, 7), laneID, sMem);
  block_comparator(value, index, 64, bfe(laneID, 8) ^ bfe(laneID, 6), laneID, sMem);
  block_comparator(value, index, 32, bfe(laneID, 8) ^ bfe(laneID, 5), laneID, sMem);
  warp_comparator(value, index, 16, bfe(laneID, 8) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 8) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 8) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 8) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 8) ^ bfe(laneID, 0));
}
#endif
__device__ void bitonic_sort_global_256(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  _VOLATILE_ float sMem[],
  int laneID
) {
  if (_TPB_ - 256 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 128, !bfe(laneID, 7), laneID, sMem);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, sMem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, sMem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
  }
}

#if _TPB_ >= 512
__device__ void bitonic_sort_512(
  float &value,
  float &index,
  _VOLATILE_ float sMem[],
  int laneID
){
  bitonic_sort_256(value, index, sMem, laneID);
  block_comparator(value, index, 256, bfe(laneID, 9) ^ bfe(laneID, 8), laneID, sMem);
  block_comparator(value, index, 128, bfe(laneID, 9) ^ bfe(laneID, 7), laneID, sMem);
  block_comparator(value, index, 64, bfe(laneID, 9) ^ bfe(laneID, 6), laneID, sMem);
  block_comparator(value, index, 32, bfe(laneID, 9) ^ bfe(laneID, 5), laneID, sMem);
  warp_comparator(value, index, 16, bfe(laneID, 9) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 9) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 9) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 9) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 9) ^ bfe(laneID, 0));
}
#endif
__device__ void bitonic_sort_global_512(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  _VOLATILE_ float sMem[],
  int laneID
) {
  if (_TPB_ - 512 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 256, !bfe(laneID, 8), laneID, sMem);
    block_comparator(value, index, 128, !bfe(laneID, 7), laneID, sMem);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, sMem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, sMem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
  }
}


#if _TPB_ >= 1024
__device__ void bitonic_sort_1024(
  float &value,
  float &index,
  _VOLATILE_ float sMem[],
  int laneID
){
  bitonic_sort_512(value, index, sMem, laneID);
  block_comparator(value, index, 512, bfe(laneID, 10) ^ bfe(laneID, 9), laneID, sMem);
  block_comparator(value, index, 256, bfe(laneID, 10) ^ bfe(laneID, 8), laneID, sMem);
  block_comparator(value, index, 128, bfe(laneID, 10) ^ bfe(laneID, 7), laneID, sMem);
  block_comparator(value, index, 64, bfe(laneID, 10) ^ bfe(laneID, 6), laneID, sMem);
  block_comparator(value, index, 32, bfe(laneID, 10) ^ bfe(laneID, 5), laneID, sMem);
  warp_comparator(value, index, 16, bfe(laneID, 10) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 10) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 10) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 10) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 10) ^ bfe(laneID, 0));
}
#endif
__device__ void bitonic_sort_global_1024(
  float &value,
  float &index,
  float otherValue,
  float otherIndex,
  _VOLATILE_ float sMem[],
  int laneID
) {
  if (_TPB_ - 1024 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 512, !bfe(laneID, 9), laneID, sMem);
    block_comparator(value, index, 256, !bfe(laneID, 8), laneID, sMem);
    block_comparator(value, index, 128, !bfe(laneID, 7), laneID, sMem);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, sMem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, sMem);
    warp_comparator(value, index, 16, !bfe(laneID, 4));
    warp_comparator(value, index, 8, !bfe(laneID, 3));
    warp_comparator(value, index, 4, !bfe(laneID, 2));
    warp_comparator(value, index, 2, !bfe(laneID, 1));
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  } else {
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
    block_comparator_noop();
  }
}

__device__ void load_precomputed_v1(
  const float *precomputed,
  _VOLATILE_ float *sMem,
  int nQuery
){
  const int tid = threadIdx.x;
  const int qid = blockIdx.x;
  if (tid < 256){
    #pragma unroll
    for (int i = 0; i < _M_; i++){
      #if _TPB_ >= 256
      int adr = (i * nQuery * _K_) + (qid * _K_) + (tid);
      sMem[i * _K_ + tid] = precomputed[adr];
      
      #else
      #pragma unroll
      for (int j = 0; j < _K_ / _TPB_; j++){
        int adr = (i * nQuery * _K_) + (qid * _K_) + (j * _TPB_ + tid);
        sMem[i * _K_ + j * _TPB_ + tid] = precomputed[adr];
      }
      #endif
    }
  }
  __syncthreads();
}

__device__ void load_precomputed_v2(
  const float *precomputed,
  _VOLATILE_ float *sMem,
  int iProbe, int nProbe
){
  const int tid = threadIdx.x;
  const int qid = blockIdx.x;
  if (tid < 256){
    #pragma unroll
    for (int i = 0; i < _M_; i++){
      #if _TPB_ >= 256
      // int adr = (i * nQuery * _K_) + (qid * _K_) + (tid);
      int adr = 
        (qid) * nProbe * _M_ * _K_ +\
        (iProbe) * _M_ * _K_ +\
        (i) * _K_ +\
        (tid);
      sMem[i * _K_ + tid] = precomputed[adr];
      
      #else
      #pragma unroll
      for (int j = 0; j < _K_ / _TPB_; j++){
        int adr = (qid) * nProbe * _M_ * _K_ +\
          (iProbe) * _M_ * _K_ +\
          (i) * _K_ +\
          (j * _TPB_ + tid);
        sMem[i * _K_ + j * _TPB_ + tid] = precomputed[adr];
      }
      #endif
    }
  }
  __syncthreads();
}

__device__ void load_precomputed_v3(
  const float* part1,
  const float* part2,
  _VOLATILE_ float *sMem,
  int iCell
){
  const int tid = threadIdx.x;
  const int qid = blockIdx.x;
  if (tid < 256){
    #pragma unroll
    for (int i = 0; i < _M_; i++){
      #if _TPB_ >= 256
      // int adr = (i * nQuery * _K_) + (qid * _K_) + (tid);
      int adr1 =\
        (qid) * _M_ * _K_ +\
        (i) * _K_ +\
        (tid);
      float precomputedValue = part1[adr1];

      int adr2 =\
        (iCell) * _M_ * _K_ +\
        (i) * _K_ +\
        (tid);
      sMem[i * _K_ + tid] = precomputedValue + part2[adr2];

      #else
      #pragma unroll
      for (int j = 0; j < _K_ / _TPB_; j++){
        int adr1 =\
          (qid) * _M_ * _K_ +\
          (i) * _K_ +\
          (j * _TPB_ + tid);
        float precomputedValue = part1[adr1];

        int adr2 =\
          (iCell) * _M_ * _K_ +\
          (i) * _K_ +\
          (j * _TPB_ + tid);
        sMem[i * _K_ + j * _TPB_ + tid] = precomputedValue + part2[adr2];
      }
      #endif
    }
  }
  __syncthreads();
}

__device__ void load_part1_to_cache(
  const float* part1,
  float part1Cache[_M_]
){
  const int tid = threadIdx.x;
  const int qid = blockIdx.x;
  if (tid < 256){
    #pragma unroll
    for (int i = 0; i < _M_; i++){
      #if _TPB_ >= 256
      int adr1 =\
        (qid) * _M_ * _K_ +\
        (i) * _K_ +\
        (tid);
      part1Cache[i] = part1[adr1];
      #endif
    }
  }
}

__device__ void load_part2_to_cache(
  const float* part2,
  float part2Cache[_M_],
  int iCell
){
  const int tid = threadIdx.x;
  const int qid = blockIdx.x;
  if (tid < 256){
    #pragma unroll
    for (int i = 0; i < _M_; i++){
      #if _TPB_ >= 256
      int adr2 =\
        (iCell) * _M_ * _K_ +\
        (i) * _K_ +\
        (tid);
      part2Cache[i] = part2[adr2];
      #endif
    }
  }
}

__device__ void store_precomputed_to_smem(
  float part1Cache[_M_],
  float part2Cache[_M_],
  _VOLATILE_ float *sMem
){
  const int tid = threadIdx.x;
  const int qid = blockIdx.x;
  __syncthreads();
  if (tid < 256){
    #pragma unroll
    for (int i = 0; i < _M_; i++){
      #if _TPB_ >= 256
      float part1Value = part1Cache[i];
      float part2Value = part2Cache[i];
      sMem[i * _K_ + tid] = part1Value + part2Value;
      #endif
    }
  }
  __syncthreads();
}

__device__ void load_consume_data(
  const uint8n_t* data,
  _VOLATILE_ float sMem[],
  float &value,
  int iN, int nData
){
  #pragma unroll
  for (int i = 0; i < _M_ / _NCS_; i++){
    uint8n_t threadData = data[(i * nData) + iN];
    float pre0 = sMem[(i * _NCS_ + 0) * _K_ + int(threadData.val[0]) ];
    float pre1 = sMem[(i * _NCS_ + 1) * _K_ + int(threadData.val[1]) ];
    float pre2 = sMem[(i * _NCS_ + 2) * _K_ + int(threadData.val[2]) ];
    float pre3 = sMem[(i * _NCS_ + 3) * _K_ + int(threadData.val[3]) ];
    value += pre0;
    value += pre1;
    value += pre2;
    value += pre3;
  }
}

__device__ void load_data(
  const uint8n_t* data,
  uint8n_t dataCache[_M_ / _NCS_],
  int iN, int nData
){
  #pragma unroll
  for (int i = 0; i < _M_ / _NCS_; i++){
    uint8n_t threadData = data[(i * nData) + iN];
    dataCache[i] = threadData;
  }
}

__device__ void consume_data(
  _VOLATILE_ float sMem[],
  uint8n_t dataCache[_M_ / _NCS_],
  float &value
){
  #pragma unroll
  for (int i = 0; i < _M_ / _NCS_; i++){
    uint8n_t threadData = dataCache[i];
    float pre0 = sMem[(i * _NCS_ + 0) * _K_ + int(threadData.val[0]) ];
    float pre1 = sMem[(i * _NCS_ + 1) * _K_ + int(threadData.val[1]) ];
    float pre2 = sMem[(i * _NCS_ + 2) * _K_ + int(threadData.val[2]) ];
    float pre3 = sMem[(i * _NCS_ + 3) * _K_ + int(threadData.val[3]) ];
    value += pre0;
    value += pre1;
    value += pre2;
    value += pre3;
  }
}

__device__ void sort(
  float &finalValue,
  float &finalIndex,
  float value,
  float index,
  _VOLATILE_ float sMem[],
  int nCandidates
){
  const int tid = threadIdx.x;
  #if _TPB_ == 32
  bitonic_sort_32(value, index, tid);

  #elif _TPB_ == 64
  bitonic_sort_64(value, index, sMem, tid);

  #elif _TPB_ == 128
  bitonic_sort_128(value, index, sMem, tid);

  #elif _TPB_ == 256
  bitonic_sort_256(value, index, sMem, tid);

  #elif _TPB_ == 512
  bitonic_sort_512(value, index, sMem, tid);

  #elif _TPB_ == 1024
  bitonic_sort_1024(value, index, sMem, tid);
  #endif
  
  switch (nCandidates){
    case 2:
      bitonic_sort_global_2(
        finalValue, finalIndex, value, index,
        tid);
        break;
    case 4:
      bitonic_sort_global_4(
        finalValue, finalIndex, value, index,
        tid);
        break;
    case 8:
      bitonic_sort_global_8(
        finalValue, finalIndex, value, index,
        tid);
        break;
    case 16:
      bitonic_sort_global_16(
        finalValue, finalIndex, value, index,
        tid);
        break;
    case 32:
      bitonic_sort_global_32(
        finalValue, finalIndex, value, index,
        tid);
        break;
    case 64:
      bitonic_sort_global_64(
        finalValue, finalIndex, value, index,
        sMem, tid);
        break;
    case 128:
      bitonic_sort_global_128(
        finalValue, finalIndex, value, index,
        sMem, tid);
        break;
    case 256:
      bitonic_sort_global_256(
        finalValue, finalIndex, value, index,
        sMem, tid);
        break;
    case 512:
      bitonic_sort_global_512(
        finalValue, finalIndex, value, index,
        sMem, tid);
        break;
    case 1024:
      bitonic_sort_global_1024(
        finalValue, finalIndex, value, index,
        sMem, tid);
        break;
  }
}

__device__ bool is_stack_empty(
  int stackSize
){
  return stackSize <= 0;
}

__device__ bool is_stack_full(
  int stackSize
){
  return stackSize >= _STACKCAP_ - 1;
}

__device__ void push_stack(
  pair stack[_STACKCAP_],
  pair newPair,
  int &stackSize
) {
  if (is_stack_full(stackSize)){
    return;
  } else {
    #pragma unroll
    for (int i = _STACKCAP_ - 1; i >= 1; i--){
      stack[i] = stack[i - 1];
    }
    stack[0] = newPair;
    stackSize ++;
  }
}

__device__ void pop_stack(
  pair stack[_STACKCAP_],
  pair &outPair,
  int &stackSize
) {
  if (is_stack_empty(stackSize)){
    return;
  } else {
    outPair = stack[0];
    #pragma unroll
    for (int i=0; i<_STACKCAP_-1; i++){
      stack[i] = stack[i+1];
    }
    stackSize--;
  }
}

__device__ void init_stack(
  pair stack[_STACKCAP_]
){
  pair emptyPair;
  emptyPair.value = -INFINITY;
  emptyPair.index = -1;
  #pragma unroll
  for (int i=0; i < _STACKCAP_; i++){
    stack[i] = emptyPair;
  }
}

extern "C"
__global__ void ivfpq_topk(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int nProbe, int nCandidates
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID

  pair stack[_STACKCAP_];
  int stackSize = 0;
  init_stack(stack);

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  load_precomputed_v1(precomputed, sMem, nQuery);
  float finalValue = -INFINITY;
  float finalIndex = -1;
  float minValue = -INFINITY;

  const ll_t threadTotalSize = totalSize[qid];
  const int nIter = (threadTotalSize + _TPB_ - 1) / _TPB_;
  int cCell = 0;
  int cCellStart = cellStart[qid * nProbe + cCell];
  int cCellSize = cellSize[qid * nProbe + cCell];
  int cCellEnd = cCellStart + cCellSize;
  int iN = cCellStart + tid;

  for (int i = 0; i < nIter; i++){
    while (iN >= cCellEnd){
      cCell ++;  // increment cell index by 1
      if (unlikely(cCell >= nProbe))
        break;
      int pCellEnd = cCellEnd;
      cCellStart = cellStart[qid * nProbe + cCell];
      cCellSize = cellSize[qid * nProbe + cCell];
      cCellEnd = cCellStart + cCellSize;
      iN = iN - pCellEnd + cCellStart;
    }
    pair newPair;
    newPair.value = -INFINITY;
    newPair.index = -1;
    int cIsEmpty = 0;
    if (likely(iN < cCellEnd)){
      newPair.value = 0.f;
      newPair.index = iN;
      cIsEmpty = isEmpty[iN];
      uint8n_t dataCache[_M_ / _NCS_];
      load_data(data, dataCache, iN, nData);
      consume_data(sMem, dataCache, newPair.value);
    }
    newPair.value = cIsEmpty == 0 ? newPair.value : -INFINITY;
    newPair.index = cIsEmpty == 0 ? newPair.index : -1;

    __syncthreads();
    float temp1, temp2;
    if (tid == 0){
      temp1 = sMem[0];
      temp2 = sMem[1];
      sMem[0] = 0;
    }
    __syncthreads();

    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (is_stack_full(stackSize)){
      pop_stack(stack, oldPair, stackSize);
      if (oldPair.value > minValue){
        sMem[0] = 1;
      }
    }

    if (newPair.value > minValue){
      push_stack(stack, newPair, stackSize);
    }
    __syncthreads();

    if (sMem[0] > 0){      
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        sMem, nCandidates
      );
      __syncthreads();
      if (tid == _TPB_ - 1){
        sMem[1] = finalValue;
      }
      __syncthreads();
      minValue = sMem[1];
      // __syncthreads();
    }
    if (tid == 0){
      sMem[0] = temp1;
      sMem[1] = temp2;
    }
    // __syncthreads();
    
    iN += _TPB_;
  }

  sMem[0] = 0;
  __syncthreads();
  for (int i=0; i<_STACKCAP_; i++){
    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (!is_stack_empty(stackSize)){
      pop_stack(stack, oldPair, stackSize);
      if (oldPair.value > minValue){
        sMem[0] = 1;
      }
    }
    __syncthreads();

    if (sMem[0] > 0){      
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        sMem, nCandidates
      );
      __syncthreads();
      sMem[0] = 0;
      if (tid == _TPB_ - 1){
        sMem[1] = finalValue;
      }
      __syncthreads();
      minValue = sMem[1];
    }
    __syncthreads();
  }

  if (_TPB_ - nCandidates <= tid){
    const int writeAddress = (qid * nCandidates) + tid - ( _TPB_ - nCandidates);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_topk_residual(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const float* __restrict__ baseSims,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int nProbe, int nCandidates
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -INFINITY;
  float finalIndex = -1;
  int cCellStart = -1;
  for (int cCell = 0; cCell < nProbe; cCell++){
    int pCellStart = cCellStart;
    cCellStart = cellStart[qid * nProbe + cCell];
    if (cCellStart == pCellStart){
      continue;
    }
    int cCellSize = cellSize[qid * nProbe + cCell];
    load_precomputed_v2(precomputed, sMem, cCell, nProbe);
    float cBaseSim = baseSims[qid * nProbe + cCell];
    int cCellEnd = cCellStart + cCellSize;
    int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
    for (int iter = 0; iter < nIter; iter++ ){
      int iN = cCellStart + iter * _TPB_ + tid;
      float value;
      float index = iN;
      int cIsEmpty = 0;
      if (cCellStart <= iN && iN < cCellEnd){
        value = cBaseSim;
        cIsEmpty = isEmpty[iN];
        uint8n_t dataCache[_M_ / _NCS_];
        load_data(data, dataCache, iN, nData);
        consume_data(sMem, dataCache, value);
      } else {
        value = -INFINITY;
      }
      value = cIsEmpty == 0 ? value : -INFINITY;
      index = cIsEmpty == 0 ? index : -1;
      
      sort(
        finalValue, finalIndex,
        value, index,
        sMem, nCandidates
      );
    }
  }

  if (_TPB_ - nCandidates <= tid){
    const int writeAddress = (qid * nCandidates) + tid - ( _TPB_ - nCandidates);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_topk_residual_precomputed(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ part1,
  const float* __restrict__ part2,
  const ll_t* __restrict__ cells,
  const float* __restrict__ baseSims,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int nProbe, int nCandidates
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  
  pair stack[_STACKCAP_];
  int stackSize = 0;
  init_stack(stack);

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -INFINITY;
  float finalIndex = -1;
  float minValue = -INFINITY;
  float part1Cache[_M_];
  float part2Cache[_M_];
  load_part1_to_cache(part1, part1Cache);

  int nCellStart = cellStart[qid * nProbe];
  int nCellSize = cellSize[qid * nProbe];
  int nCellEnd = nCellStart + nCellSize;
  int iCell = cells[qid * nProbe];
  bool nCellRepeated = false;
  bool cCellRepeated = false;
  load_part2_to_cache(part2, part2Cache, iCell);

  for (int cCell = 0; cCell < nProbe; cCell++){
    int cCellStart = nCellStart;
    int cCellSize = nCellSize;
    int cCellEnd = nCellEnd;
    if (!cCellRepeated){
      store_precomputed_to_smem(part1Cache, part2Cache, sMem);
    }

    if (cCell < nProbe - 1){
      int tCellStart = cellStart[qid * nProbe + cCell + 1];
      if (tCellStart != cCellStart){
        nCellStart = tCellStart;
        nCellSize = cellSize[qid * nProbe + cCell + 1];
        nCellEnd = nCellStart + nCellSize;
        iCell = cells[qid * nProbe + cCell + 1];
        load_part2_to_cache(part2, part2Cache, iCell);
        nCellRepeated = false;
      } else {
        nCellRepeated = true;
      }
    }
    if (cCellRepeated){
      cCellRepeated = nCellRepeated;
      continue;
    }
    cCellRepeated = nCellRepeated;
    float cBaseSim = baseSims[qid * nProbe + cCell];
    int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
    for (int iter = 0; iter < nIter; iter++ ){
      int iN = cCellStart + iter * _TPB_ + tid;
      pair newPair;
      newPair.value = -INFINITY;
      newPair.index = -1;
      int cIsEmpty = 0;
      if (iN < cCellEnd){
        newPair.value = cBaseSim;
        newPair.index = iN;
        cIsEmpty = isEmpty[iN];
        uint8n_t dataCache[_M_ / _NCS_];
        load_data(data, dataCache, iN, nData);
        consume_data(sMem, dataCache, newPair.value);
      }

      newPair.value = cIsEmpty == 0 ? newPair.value : -INFINITY;
      newPair.index = cIsEmpty == 0 ? newPair.index : -1;
      
      __syncthreads();
      float temp1, temp2;
      if (tid == 0){
        temp1 = sMem[0];
        temp2 = sMem[1];
        sMem[0] = 0;
      }
      __syncthreads();

      pair oldPair;
      oldPair.value = -INFINITY;
      oldPair.index = -1;
      if (is_stack_full(stackSize)){
        pop_stack(stack, oldPair, stackSize);
        if (oldPair.value > minValue){
          sMem[0] = 1;
        }
      }
      if (newPair.value > minValue){
        push_stack(stack, newPair, stackSize);
      }
      __syncthreads();
      if (sMem[0] > 0){
        sort(
          finalValue, finalIndex,
          oldPair.value, oldPair.index,
          sMem, nCandidates
        );
        __syncthreads();
        if (tid == _TPB_ - 1){
          sMem[1] = finalValue;
        }
        __syncthreads();
        minValue = sMem[1];
      }
      if (tid == 0){
        sMem[0] = temp1;
        sMem[1] = temp2;
      }
    }
  }
  sMem[0] = 0;
  __syncthreads();
  for (int i=0; i < _STACKCAP_; i++){
    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (!is_stack_empty(stackSize)){
      pop_stack(stack, oldPair, stackSize);
      if (oldPair.value > minValue){
        sMem[0] = 1;
      }
    }
    __syncthreads();

    if (sMem[0] > 0){
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        sMem, nCandidates
      );
      __syncthreads();
      sMem[0] = 0;
      if (tid == _TPB_ - 1){
        sMem[1] = finalValue;
      }
      __syncthreads();
      minValue = sMem[1];
    }
    __syncthreads();
  }

  if (_TPB_ - nCandidates <= tid){
    const int writeAddress = (qid * nCandidates) + tid - ( _TPB_ - nCandidates);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_topk_smart_probing(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  const ll_t* __restrict__ nProbeList,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int maxNProbe, int nCandidates
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  const int nProbe = nProbeList[qid];

  pair stack[_STACKCAP_];
  int stackSize = 0;
  init_stack(stack);

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  load_precomputed_v1(precomputed, sMem, nQuery);
  float finalValue = -INFINITY;
  float finalIndex = -1;
  float minValue = -INFINITY;
  const ll_t threadTotalSize = totalSize[qid];
  const int nIter = (threadTotalSize + _TPB_ - 1) / _TPB_;
  int cCell = 0;
  int cCellStart = cellStart[qid * maxNProbe + cCell];
  int cCellSize = cellSize[qid * maxNProbe + cCell];
  int cCellEnd = cCellStart + cCellSize;
  int iN = cCellStart + tid;

  for (int i = 0; i < nIter; i++){
    while (iN >= cCellEnd){
      cCell ++;  // increment cell index by 1
      if (cCell >= nProbe)
        break;
      int pCellEnd = cCellEnd;
      int pCellStart = cCellStart;
      cCellStart = cellStart[qid * maxNProbe + cCell];
      if (cCellStart == pCellStart){
        continue;
      }
      cCellSize = cellSize[qid * maxNProbe + cCell];
      cCellEnd = cCellStart + cCellSize;
      iN = iN - pCellEnd + cCellStart;
    }
    pair newPair;
    newPair.value = -INFINITY;
    newPair.index = -1;
    int cIsEmpty = 0;
    if (likely(iN < cCellEnd)){
      newPair.value = 0.f;
      newPair.index = iN;
      cIsEmpty = isEmpty[iN];
      uint8n_t dataCache[_M_ / _NCS_];
      load_data(data, dataCache, iN, nData);
      consume_data(sMem, dataCache, newPair.value);
    }
    newPair.value = cIsEmpty == 0 ? newPair.value : -INFINITY;
    newPair.index = cIsEmpty == 0 ? newPair.index : -1;

    __syncthreads();
    float temp1, temp2;
    if (tid == 0){
      temp1 = sMem[0];
      temp2 = sMem[1];
      sMem[0] = 0;
    }
    __syncthreads();

    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (is_stack_full(stackSize)){
      pop_stack(stack, oldPair, stackSize);
      if (oldPair.value > minValue){
        sMem[0] = 1;
      }
    }

    if (newPair.value > minValue){
      push_stack(stack, newPair, stackSize);
    }
    __syncthreads();

    if (sMem[0] > 0){      
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        sMem, nCandidates
      );
      __syncthreads();
      if (tid == _TPB_ - 1){
        sMem[1] = finalValue;
      }
      __syncthreads();
      minValue = sMem[1];
      // __syncthreads();
    }
    if (tid == 0){
      sMem[0] = temp1;
      sMem[1] = temp2;
    }
    iN += _TPB_;
  }

  sMem[0] = 0;
  __syncthreads();
  #pragma unroll
  for (int i=0; i<_STACKCAP_; i++){
    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (!is_stack_empty(stackSize)){
      pop_stack(stack, oldPair, stackSize);
      if (oldPair.value > minValue){
        sMem[0] = 1;
      }
    }
    __syncthreads();

    if (sMem[0] > 0){      
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        sMem, nCandidates
      );
      __syncthreads();
      sMem[0] = 0;
      if (tid == _TPB_ - 1){
        sMem[1] = finalValue;
      }
      __syncthreads();
      minValue = sMem[1];
    }
    __syncthreads();
  }

  if (_TPB_ - nCandidates <= tid){
    const int writeAddress = (qid * nCandidates) + tid - ( _TPB_ - nCandidates);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_topk_residual_smart_probing(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const float* __restrict__ baseSims,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  const ll_t* __restrict__ nProbeList,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int maxNProbe, int nCandidates
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  const int nProbe = nProbeList[qid];

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -INFINITY;
  float finalIndex = -1;
  int cCellStart = -1;
  for (int cCell = 0; cCell < nProbe; cCell++){
    int pCellStart = cCellStart;
    cCellStart = cellStart[qid * maxNProbe + cCell];
    if (cCellStart == pCellStart){
      continue;
    }
    int cCellSize = cellSize[qid * maxNProbe + cCell];
    load_precomputed_v2(precomputed, sMem, cCell, maxNProbe);
    float cBaseSim = baseSims[qid * maxNProbe + cCell];
    int cCellEnd = cCellStart + cCellSize;
    int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
    for (int iter = 0; iter < nIter; iter++ ){
      int iN = cCellStart + iter * _TPB_ + tid;
      float value;
      float index = iN;
      int cIsEmpty = 0;
      if (cCellStart <= iN && iN < cCellEnd){
        value = cBaseSim;
        cIsEmpty = isEmpty[iN];
        uint8n_t dataCache[_M_ / _NCS_];
        load_data(data, dataCache, iN, nData);
        consume_data(sMem, dataCache, value);
      } else {
        value = -INFINITY;
      }
      value = cIsEmpty == 0 ? value : -INFINITY;
      index = cIsEmpty == 0 ? index : -1;
      
      sort(
        finalValue, finalIndex,
        value, index,
        sMem, nCandidates
      );
    }
  }

  if (_TPB_ - nCandidates <= tid){
    const int writeAddress = (qid * nCandidates) + tid - ( _TPB_ - nCandidates);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_topk_residual_precomputed_smart_probing(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ part1,
  const float* __restrict__ part2,
  const ll_t* __restrict__ cells,
  const float* __restrict__ baseSims,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  const ll_t* __restrict__ nProbeList,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int maxNProbe, int nCandidates
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  const int nProbe = nProbeList[qid];

  pair stack[_STACKCAP_];
  int stackSize = 0;
  init_stack(stack);

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -INFINITY;
  float finalIndex = -1;
  float minValue = -INFINITY;
  float part1Cache[_M_];
  float part2Cache[_M_];
  load_part1_to_cache(part1, part1Cache);

  int nCellStart = cellStart[qid * maxNProbe];
  int nCellSize = cellSize[qid * maxNProbe];
  int nCellEnd = nCellStart + nCellSize;
  int iCell = cells[qid * maxNProbe];
  bool nCellRepeated = false;
  bool cCellRepeated = false;
  load_part2_to_cache(part2, part2Cache, iCell);

  for (int cCell = 0; cCell < nProbe; cCell++){
    int cCellStart = nCellStart;
    int cCellSize = nCellSize;
    int cCellEnd = nCellEnd;
    if (!cCellRepeated){
      store_precomputed_to_smem(part1Cache, part2Cache, sMem);
    }

    if (cCell < nProbe - 1){
      int tCellStart = cellStart[qid * maxNProbe + cCell + 1];
      if (tCellStart != cCellStart){
        nCellStart = tCellStart;
        nCellSize = cellSize[qid * maxNProbe + cCell + 1];
        nCellEnd = nCellStart + nCellSize;
        iCell = cells[qid * maxNProbe + cCell + 1];
        load_part2_to_cache(part2, part2Cache, iCell);
        nCellRepeated = false;
      } else {
        nCellRepeated = true;
      }
    }
    if (cCellRepeated){
      cCellRepeated = nCellRepeated;
      continue;
    }
    cCellRepeated = nCellRepeated;
    float cBaseSim = baseSims[qid * maxNProbe + cCell];
    int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
    for (int iter = 0; iter < nIter; iter++ ){
      int iN = cCellStart + iter * _TPB_ + tid;
      pair newPair;
      newPair.value = -INFINITY;
      newPair.index = -1;
      int cIsEmpty = 0;
      if (iN < cCellEnd){
        newPair.value = cBaseSim;
        newPair.index = iN;
        cIsEmpty = isEmpty[iN];
        uint8n_t dataCache[_M_ / _NCS_];
        load_data(data, dataCache, iN, nData);
        consume_data(sMem, dataCache, newPair.value);
      }

      newPair.value = cIsEmpty == 0 ? newPair.value : -INFINITY;
      newPair.index = cIsEmpty == 0 ? newPair.index : -1;
      
      __syncthreads();
      float temp1, temp2;
      if (tid == 0){
        temp1 = sMem[0];
        temp2 = sMem[1];
        sMem[0] = 0;
      }
      __syncthreads();

      pair oldPair;
      oldPair.value = -INFINITY;
      oldPair.index = -1;
      if (is_stack_full(stackSize)){
        pop_stack(stack, oldPair, stackSize);
        if (oldPair.value > minValue){
          sMem[0] = 1;
        }
      }
      if (newPair.value > minValue){
        push_stack(stack, newPair, stackSize);
      }
      __syncthreads();
      if (sMem[0] > 0){
        sort(
          finalValue, finalIndex,
          oldPair.value, oldPair.index,
          sMem, nCandidates
        );
        __syncthreads();
        if (tid == _TPB_ - 1){
          sMem[1] = finalValue;
        }
        __syncthreads();
        minValue = sMem[1];
      }
      if (tid == 0){
        sMem[0] = temp1;
        sMem[1] = temp2;
      }
    }
  }

  sMem[0] = 0;
  __syncthreads();
  for (int i=0; i < _STACKCAP_; i++){
    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (!is_stack_empty(stackSize)){
      pop_stack(stack, oldPair, stackSize);
      if (oldPair.value > minValue){
        sMem[0] = 1;
      }
    }
    __syncthreads();

    if (sMem[0] > 0){
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        sMem, nCandidates
      );
      __syncthreads();
      sMem[0] = 0;
      if (tid == _TPB_ - 1){
        sMem[1] = finalValue;
      }
      __syncthreads();
      minValue = sMem[1];
    }
    __syncthreads();
  }

  if (_TPB_ - nCandidates <= tid){
    const int writeAddress = (qid * nCandidates) + tid - ( _TPB_ - nCandidates);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}