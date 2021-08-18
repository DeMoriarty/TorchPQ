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
  bool condition = value < otherValue == direction;
  index = condition ? otherIndex : index;
  value = condition ? otherValue : value;
}

__device__ __forceinline__ void block_comparator(
  float &value,
  int &index,
  const int stride,
  const int direction,
  const int laneID,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_]
){
  valSmem[laneID] = value;
  idxSmem[laneID] = index;
  __syncthreads();

  float otherValue = valSmem[laneID ^ stride];
  float otherIndex = idxSmem[laneID ^ stride];
  __syncthreads();

  bool condition = value < otherValue == direction;
  value = condition ? otherValue : value;
  index = condition ? otherIndex : index;
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
  warp_comparator(value, index, 2, bfe(laneID, 2) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 2) ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_8(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_4(value, index, laneID);
  warp_comparator(value, index, 4, bfe(laneID, 3) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 3) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 3) ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_16(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_8(value, index, laneID);
  warp_comparator(value, index, 8, bfe(laneID, 4) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 4) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 4) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 4) ^ bfe(laneID, 0));
}

__device__ __forceinline__ void bitonic_sort_32(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_16(value, index, laneID);
  warp_comparator(value, index, 16, bfe(laneID, 5) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 5) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 5) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 5) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 5) ^ bfe(laneID, 0));
}