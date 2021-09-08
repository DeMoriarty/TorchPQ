#include "cuda_fp16.h"

#define _VOLATILE_ 
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define isnan(x) ( x != x )
#define N_WARPS _TPB_/32
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

typedef long long ll_t;
typedef struct __builtin_align__(8)
{
  float value;
  int index;
} pair;

#if (__CUDA_ARCH__ < 700)
__device__ void __nanosleep(unsigned int ns){
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < ns)
  {
    clock_offset = clock() - start_clock;
  }
}
#endif 


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
  int &index,
  const int stride,
  const int direction
){
  const float otherValue = __shfl_xor_sync(0xFFFFFFFF, value, stride);
  const int otherIndex = __shfl_xor_sync(0xFFFFFFFF, index, stride);
  // bool condition = value < otherValue == direction;
  // index = condition ? otherIndex : index;
  // value = condition ? otherValue : value;
  if (value < otherValue == direction){
    index = otherIndex;
    value = otherValue;
  }
}

__device__ __forceinline__ void thread_comparator(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  const int direction
){
  bool condition = value > otherValue == direction;
  if (condition){
    value = otherValue;
    index = otherIndex;
  }
}

__device__ __forceinline__ void bitonic_sort_2(
  float &value,
  int &index,
  int laneID
){
  warp_comparator(value, index, 1, bfe(laneID, 1) ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_4(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_2(value, index, laneID);
  unsigned int bfe_2 = bfe(laneID, 2);
  warp_comparator(value, index, 2, bfe_2 ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe_2 ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_8(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_4(value, index, laneID);
  unsigned int bfe_3 = bfe(laneID, 3);
  warp_comparator(value, index, 4, bfe_3 ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe_3 ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe_3 ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_16(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_8(value, index, laneID);
  unsigned int bfe_4 = bfe(laneID, 4);
  warp_comparator(value, index, 8, bfe_4 ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe_4 ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe_4 ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe_4 ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_32(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_16(value, index, laneID);
  unsigned int bfe_5 = bfe(laneID, 5);
  warp_comparator(value, index, 16, bfe_5 ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe_5 ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe_5 ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe_5 ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe_5 ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_global_1(
  float &value,
  int &index,
  float otherValue,
  int otherIndex
) {
  thread_comparator(value, index, otherValue, otherIndex, 0);
}

__device__ __forceinline__ void bitonic_sort_global_2(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  int laneID
) {
  thread_comparator(value, index, otherValue, otherIndex, 0);
  warp_comparator(value, index, 1, !bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_global_4(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  int laneID
) {
  thread_comparator(value, index, otherValue, otherIndex, 0);
  warp_comparator(value, index, 2, !bfe(laneID, 1));
  warp_comparator(value, index, 1, !bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_global_8(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  int laneID
) {
  thread_comparator(value, index, otherValue, otherIndex, 0);
  warp_comparator(value, index, 4, !bfe(laneID, 2));
  warp_comparator(value, index, 2, !bfe(laneID, 1));
  warp_comparator(value, index, 1, !bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_global_16(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  int laneID
) {
  thread_comparator(value, index, otherValue, otherIndex, 0);
  warp_comparator(value, index, 8, !bfe(laneID, 3));
  warp_comparator(value, index, 4, !bfe(laneID, 2));
  warp_comparator(value, index, 2, !bfe(laneID, 1));
  warp_comparator(value, index, 1, !bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_global_32(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  int laneID
) {
    thread_comparator(value, index, otherValue, otherIndex, 0);
  warp_comparator(value, index, 16, !bfe(laneID, 4));
  warp_comparator(value, index, 8, !bfe(laneID, 3));
  warp_comparator(value, index, 4, !bfe(laneID, 2));
  warp_comparator(value, index, 2, !bfe(laneID, 1));
  warp_comparator(value, index, 1, !bfe(laneID, 0));
}

__device__ __forceinline__ bool is_queue_full(
  int queueFront,
  int queueRear
){
  // return ((queueFront - queueRear) == 1 || (queueFront == 0 && queueRear == _QCAP_ - 1));
  return (queueRear + 1) % _QCAP_ == queueFront;
}

__device__ __forceinline__ bool is_queue_empty(
  int queueFront,
  int queueRear
){
  return queueFront == -1;
}

__device__ __forceinline__ void push_queue(
  _VOLATILE_ pair queueSmem[N_WARPS][32][_QCAP_],
  pair newPair,
  int &queueFront,
  int &queueRear,
  int wx,
  int wy
) {
  // const int tid = threadIdx.x;
  if (is_queue_full(queueFront, queueRear)){
    return;
  } else if (is_queue_empty(queueFront, queueRear)){
    queueFront = 0;
    queueRear = 0;
    queueSmem[wy][wx][queueRear] = newPair;
  } else {
    queueRear = (queueRear + 1) % _QCAP_;
    queueSmem[wy][wx][queueRear] = newPair;
  }
}

__device__ __forceinline__ void pop_queue(
  _VOLATILE_ pair queueSmem[N_WARPS][32][_QCAP_],
  pair &oldPair,
  int &queueFront,
  int &queueRear,
  int wx,
  int wy
) {
  if (is_queue_empty(queueFront, queueRear)){
    return;
  } else if (queueFront == queueRear){
    pair poppedPair = queueSmem[wy][wx][queueFront];
    oldPair.value = poppedPair.value;
    oldPair.index = poppedPair.index;

    queueFront = -1;
    queueRear = -1;
  } else {
    pair poppedPair = queueSmem[wy][wx][queueFront];
    oldPair.value = poppedPair.value;
    oldPair.index = poppedPair.index;
    
    //oldPair = queueSmem[tid][queueFront];
    queueFront = (queueFront + 1) % _QCAP_;
  }
}

__device__ __forceinline__ void push_pop_queue(
  _VOLATILE_ pair queueSmem[N_WARPS][32][_QCAP_],
  pair newPair,
  pair &oldPair,
  int &queueFront,
  int &queueRear,
  int wx,
  int wy
) {
  const int tid = threadIdx.x;
  if (is_queue_empty(queueFront, queueRear)){
    return;
  } else if (queueFront == queueRear){
    oldPair = queueSmem[wy][wx][queueFront];
    queueSmem[wy][wx][queueRear] = newPair;
  } else {
    oldPair = queueSmem[wy][wx][queueFront];
    queueFront = (queueFront + 1) % _QCAP_;
    queueRear = (queueRear + 1) % _QCAP_;
    queueSmem[wy][wx][queueRear] = newPair;
  }
}

__device__ __forceinline__ void init_queue(
  _VOLATILE_ pair queueSmem[N_WARPS][32][_QCAP_],
  const int wx,
  const int wy
){
  pair emptyPair;
  emptyPair.value = -INFINITY;
  emptyPair.index = -1;
  #pragma unroll
  for (int i=0; i<_QCAP_; i++){
    queueSmem[wy][wx][i] = emptyPair;
  }
}

__device__ __forceinline__ void sort_(
  float &finalValue,
  int &finalIndex,
  float value,
  int index,
  int K
){
  int tid = threadIdx.x;
  int wx;
  // int wx = tid % 32;
  // int wy = tid / 32;
  // #if _TPB_ == 32
  // bitonic_sort_32(value, index, wx);
  switch (K){
    case 1:
      bitonic_sort_global_1(
        finalValue, finalIndex,
        value, index);
      break;
    case 2:
      wx = tid % 2;
      bitonic_sort_2(value, index, wx);
      bitonic_sort_global_2(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 4:
      wx = tid % 4;
      bitonic_sort_4(value, index, wx);
      bitonic_sort_global_4(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 8:
      wx = tid % 8;
      bitonic_sort_8(value, index, wx);
      bitonic_sort_global_8(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 16:
      wx = tid % 16;
      bitonic_sort_16(value, index, wx);
      bitonic_sort_global_16(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 32:
      wx = tid % 32;
      bitonic_sort_32(value, index, wx);
      bitonic_sort_global_32(
        finalValue, finalIndex,
        value, index, wx);
      break;
  }
}

__device__ __forceinline__ void sort(
  float &finalValue,
  int &finalIndex,
  float value,
  int index,
  int K
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  // #if _TPB_ == 32
  bitonic_sort_32(value, index, wx);
  switch (K){
    case 1:
      bitonic_sort_global_1(
        finalValue, finalIndex,
        value, index);
      break;

    case 2:
      bitonic_sort_global_2(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 4:
      bitonic_sort_global_4(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 8:
      bitonic_sort_global_8(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 16:
      bitonic_sort_global_16(
        finalValue, finalIndex,
        value, index, wx);
      break;
    case 32:
      bitonic_sort_global_32(
        finalValue, finalIndex,
        value, index, wx);
      break;
  }
}

__device__ __forceinline__ void prefetch(
  const float* mat,
  const int i,
  const int iM,
  const int wx,
  const int N
){
  int iN = (
    i * 32
    + wx
  );
  if (likely(iN < N)){
    const float* address = mat + (iM) * N+ iN;
    // asm("prefetch_batched.global.L1 [%0];" :: "l"(address) );
    asm("prefetchu.L1 [%0];" :: "l"(address) );
  }
}

__device__ __forceinline__ void prefetch_batched(
  const float* mat,
  const int i,
  const int iM,
  const int wx,
  const int N
){
  #pragma unroll
  for (int j=0; j<_TN_; j++){
    int iN = (
      i * _TN_ * 32 
      + j * 32
      + wx
    );
    if (likely(iN < N)){
      const float* address = mat + (iM) * N+ iN;
      // asm("prefetch_batched.global.L1 [%0];" :: "l"(address) );
      asm("prefetchu.L1 [%0];" :: "l"(address) );
    }
  }

}

__device__ __forceinline__ void prefetch_batched_fp16(
  const __half* mat,
  const int i,
  const int iM,
  const int wx,
  const int N
){
  #pragma unroll
  for (int j=0; j<_TN_; j++){
    int iN = (
      i * _TN_ * 32 
      + j * 32
      + wx
    );
    if (likely(iN < N)){
      const __half* address = mat + (iM) * N+ iN;
      // asm("prefetch_batched.global.L1 [%0];" :: "l"(address) );
      asm("prefetchu.L1 [%0];" :: "l"(address) );
    }
  }

}

__device__ __forceinline__ void load_buffer_batched(
  const float* mat,
  pair buffer[_TN_],
  const int i,
  const int iM,
  const int wx,
  const int N
){
  const int tid = threadIdx.x;
  #pragma unroll
  for (int j=0; j<_TN_; j++){
    int iN = (
      i * _TN_ * 32 
      + j * 32
      + wx
    );
    if (likely(iN < N)){
      buffer[j].index = iN;
      buffer[j].value = mat[
        (iM) * N
        + iN
      ];
    } else {
      buffer[j].value = -INFINITY;
      buffer[j].index = -1;
    }
  }
}

__device__ __forceinline__ void load_buffer_batched_fp16(
  const __half* mat,
  pair buffer[_TN_],
  const int i,
  const int iM,
  const int wx,
  const int N
){
  const int tid = threadIdx.x;
  #pragma unroll
  for (int j=0; j<_TN_; j++){
    int iN = (
      i * _TN_ * 32 
      + j * 32
      + wx
    );
    if (likely(iN < N)){
      buffer[j].index = iN;
      buffer[j].value = __half2float(mat[
        (iM) * N
        + iN
      ]);
    } else {
      buffer[j].value = -INFINITY;
      buffer[j].index = -1;
    }
  }
}

__device__ __forceinline__ void arr2arr(
  pair src[_TN_],
  pair tar[_TN_]
){
  #pragma unroll
  for (int i=0; i<_TN_; i++){
    tar[i] = src[i];
  }
}

extern "C"
__global__ void top1_select(
   const float* __restrict__ mat,
   float* __restrict__ gValue,
   ll_t* __restrict__ gIndex,
   int M, int N, int K
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  // const ll_t iM = blockIdx.x;
  const int mStart = blockIdx.x * N_WARPS;
  const int iM = mStart + wy;

  pair finalPair;
  finalPair.value = -INFINITY;
  finalPair.index = -1;

  pair working[_TN_];
  prefetch_batched(mat, 0, iM, wx, N);
  const int nIter = (N + 32 * _TN_ - 1) / (32 * _TN_);
  for (int i=0; i < nIter; i++){
    if (i + 1 < nIter){
      prefetch_batched(mat, i + 1, iM, wx, N);
    }
    load_buffer_batched(mat, working, i, iM, wx, N);
    #pragma unroll
    for (int j=0; j < _TN_; j++){
      pair newPair = working[j];
      if (newPair.value > finalPair.value){
        finalPair = newPair;
      }
    }
  }
  // sort(
  //   finalPair.value, finalPair.index,
  //   finalPair.value, finalPair.index,
  //   K
  // );
  bitonic_sort_32(
    finalPair.value,
    finalPair.index,
    wx
  );

  // last K threads write their finalValue and finalIndex to gValue and gIndex
  if (32 - K <= wx){
    const int writeAddress = (iM * K) + wx - (32 - K);
    gValue[writeAddress] = finalPair.value;
    gIndex[writeAddress] = ll_t(finalPair.index);
  }
}


extern "C"
__global__ void top1_select_fp16(
   const __half* __restrict__ mat,
   __half* __restrict__ gValue,
   ll_t* __restrict__ gIndex,
   int M, int N, int K
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  // const ll_t iM = blockIdx.x;
  const int mStart = blockIdx.x * N_WARPS;
  const int iM = mStart + wy;

  pair finalPair;
  finalPair.value = -INFINITY;
  finalPair.index = -1;

  pair working[_TN_];
  prefetch_batched_fp16(mat, 0, iM, wx, N);
  const int nIter = (N + 32 * _TN_ - 1) / (32 * _TN_);
  for (int i=0; i < nIter; i++){
    if (i + 1 < nIter){
      prefetch_batched_fp16(mat, i + 1, iM, wx, N);
    }
    load_buffer_batched_fp16(mat, working, i, iM, wx, N);
    #pragma unroll
    for (int j=0; j < _TN_; j++){
      pair newPair = working[j];
      if (newPair.value > finalPair.value){
        finalPair = newPair;
      }
    }
  }
  // sort(
  //   finalPair.value, finalPair.index,
  //   finalPair.value, finalPair.index,
  //   K
  // );
  bitonic_sort_32(
    finalPair.value,
    finalPair.index,
    wx
  );

  // last K threads write their finalValue and finalIndex to gValue and gIndex
  if (32 - K <= wx){
    const int writeAddress = (iM * K) + wx - (32 - K);
    gValue[writeAddress] = __float2half(finalPair.value);
    gIndex[writeAddress] = ll_t(finalPair.index);
  }
}