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

__device__ __forceinline__ void block_comparator_noop(
){
  __syncthreads();
  __syncthreads();
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

__device__ void bitonic_sort_2(
  float &value,
  int &index,
  int laneID
){
  warp_comparator(value, index, 1, bfe(laneID, 1) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_4(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_2(value, index, laneID);
  warp_comparator(value, index, 2, bfe(laneID, 2) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 2) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_8(
  float &value,
  int &index,
  int laneID
){
  bitonic_sort_4(value, index, laneID);
  warp_comparator(value, index, 4, bfe(laneID, 3) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 3) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 3) ^ bfe(laneID, 0));
}

__device__ void bitonic_sort_16(
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

__device__ void bitonic_sort_32(
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

__device__ void bitonic_sort_global_2(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  int laneID
) {
  if (_TPB_ - 32 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    warp_comparator(value, index, 1, !bfe(laneID, 0));
  }
}

__device__ void bitonic_sort_global_4(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
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
  int &index,
  float otherValue,
  int otherIndex,
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
  int &index,
  float otherValue,
  int otherIndex,
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
  int &index,
  float otherValue,
  int otherIndex,
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
  int &index,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
){
  bitonic_sort_32(value, index, laneID);
  block_comparator(value, index, 32, bfe(laneID, 6) ^ bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 6) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 6) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 6) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 6) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 6) ^ bfe(laneID, 0));
}
#endif
__device__ void bitonic_sort_global_64(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
) {
  if (_TPB_ - 64 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, valSmem, idxSmem);
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
  int &index,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
){
  bitonic_sort_64(value, index, valSmem, idxSmem, laneID);
  block_comparator(value, index, 64, bfe(laneID, 7) ^ bfe(laneID, 6), laneID, valSmem, idxSmem);
  block_comparator(value, index, 32, bfe(laneID, 7) ^ bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 7) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 7) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 7) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 7) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 7) ^ bfe(laneID, 0));
}
#endif

__device__ void bitonic_sort_global_128(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
) {
  if (_TPB_ - 128 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, valSmem, idxSmem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, valSmem, idxSmem);
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
  int &index,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
){
  bitonic_sort_128(value, index, valSmem, idxSmem, laneID);
  block_comparator(value, index, 128, bfe(laneID, 8) ^ bfe(laneID, 7), laneID, valSmem, idxSmem);
  block_comparator(value, index, 64, bfe(laneID, 8) ^ bfe(laneID, 6), laneID, valSmem, idxSmem);
  block_comparator(value, index, 32, bfe(laneID, 8) ^ bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 8) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 8) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 8) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 8) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 8) ^ bfe(laneID, 0));
}
#endif

__device__ void bitonic_sort_global_256(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
) {
  if (_TPB_ - 256 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 128, !bfe(laneID, 7), laneID, valSmem, idxSmem);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, valSmem, idxSmem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, valSmem, idxSmem);
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
  int &index,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
){
  bitonic_sort_256(value, index, valSmem, idxSmem, laneID);
  block_comparator(value, index, 256, bfe(laneID, 9) ^ bfe(laneID, 8), laneID, valSmem, idxSmem);
  block_comparator(value, index, 128, bfe(laneID, 9) ^ bfe(laneID, 7), laneID, valSmem, idxSmem);
  block_comparator(value, index, 64, bfe(laneID, 9) ^ bfe(laneID, 6), laneID, valSmem, idxSmem);
  block_comparator(value, index, 32, bfe(laneID, 9) ^ bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 9) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 9) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 9) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 9) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 9) ^ bfe(laneID, 0));
}

#endif
__device__ void bitonic_sort_global_512(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
) {
  if (_TPB_ - 512 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 256, !bfe(laneID, 8), laneID, valSmem, idxSmem);
    block_comparator(value, index, 128, !bfe(laneID, 7), laneID, valSmem, idxSmem);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, valSmem, idxSmem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, valSmem, idxSmem);
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
  int &index,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
){
  bitonic_sort_512(value, index, valSmem, idxSmem, laneID);
  block_comparator(value, index, 512, bfe(laneID, 10) ^ bfe(laneID, 9), laneID, valSmem, idxSmem);
  block_comparator(value, index, 256, bfe(laneID, 10) ^ bfe(laneID, 8), laneID, valSmem, idxSmem);
  block_comparator(value, index, 128, bfe(laneID, 10) ^ bfe(laneID, 7), laneID, valSmem, idxSmem);
  block_comparator(value, index, 64, bfe(laneID, 10) ^ bfe(laneID, 6), laneID, valSmem, idxSmem);
  block_comparator(value, index, 32, bfe(laneID, 10) ^ bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 10) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 10) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 10) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 10) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 10) ^ bfe(laneID, 0));
}
#endif

__device__ void bitonic_sort_global_1024(
  float &value,
  int &index,
  float otherValue,
  int otherIndex,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int laneID
) {
  if (_TPB_ - 256 <= threadIdx.x){
    thread_comparator(value, index, otherValue, otherIndex, 0);
    block_comparator(value, index, 512, !bfe(laneID, 9), laneID, valSmem, idxSmem);
    block_comparator(value, index, 256, !bfe(laneID, 8), laneID, valSmem, idxSmem);
    block_comparator(value, index, 128, !bfe(laneID, 7), laneID, valSmem, idxSmem);
    block_comparator(value, index, 64, !bfe(laneID, 6), laneID, valSmem, idxSmem);
    block_comparator(value, index, 32, !bfe(laneID, 5), laneID, valSmem, idxSmem);
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

__device__ __inline__ bool is_queue_full(
  int queueFront,
  int queueRear
){
  return ((queueFront - queueRear) == 1 || (queueFront == 0 && queueRear == _QCAP_ - 1));
  //return (queueRear + 1) % _QCAP_ == queueFront;
}

__device__ __inline__ bool is_queue_empty(
  int queueFront,
  int queueRear
){
  return queueFront == -1;
}

__device__ void push_queue(
  _VOLATILE_ pair queueSmem[_TPB_][_QCAP_],
  pair newPair,
  int &queueFront,
  int &queueRear
) {
  const int tid = threadIdx.x;
  if (is_queue_full(queueFront, queueRear)){
    return;
  } else if (is_queue_empty(queueFront, queueRear)){
    queueFront = 0;
    queueRear = 0;
    queueSmem[tid][queueRear] = newPair;
  } else {
    queueRear = (queueRear + 1) % _QCAP_;
    queueSmem[tid][queueRear] = newPair;
  }
}

__device__ void pop_queue(
  _VOLATILE_ pair queueSmem[_TPB_][_QCAP_],
  pair &oldPair,
  int &queueFront,
  int &queueRear
) {
  const int tid = threadIdx.x;
  if (is_queue_empty(queueFront, queueRear)){
    return;
  } else if (queueFront == queueRear){
    pair poppedPair = queueSmem[tid][queueFront];
    oldPair.value = poppedPair.value;
    oldPair.index = poppedPair.index;

    queueFront = -1;
    queueRear = -1;
  } else {
    pair poppedPair = queueSmem[tid][queueFront];
    oldPair.value = poppedPair.value;
    oldPair.index = poppedPair.index;
    
    //oldPair = queueSmem[tid][queueFront];
    queueFront = (queueFront + 1) % _QCAP_;
  }
}

__device__ void push_pop_queue(
  _VOLATILE_ pair queueSmem[_TPB_][_QCAP_],
  pair newPair,
  pair &oldPair,
  int &queueFront,
  int &queueRear
) {
  const int tid = threadIdx.x;
  if (is_queue_empty(queueFront, queueRear)){
    return;
  } else if (queueFront == queueRear){
    oldPair = queueSmem[tid][queueFront];
    queueSmem[tid][queueRear] = newPair;
  } else {
    oldPair = queueSmem[tid][queueFront];
    queueFront = (queueFront + 1) % _QCAP_;
    queueRear = (queueRear + 1) % _QCAP_;
    queueSmem[tid][queueRear] = newPair;
  }
}

__device__ void init_queue(
  _VOLATILE_ pair queueSmem[_TPB_][_QCAP_]
){
  const int tid = threadIdx.x;
  pair emptyPair;
  emptyPair.value = -INFINITY;
  emptyPair.index = -1;
  #pragma unroll
  for (int i=0; i<_QCAP_; i++){
    queueSmem[tid][i] = emptyPair;
  }
}

__device__ void sort(
  float &finalValue,
  int &finalIndex,
  float value,
  int index,
  _VOLATILE_ float valSmem[_TPB_],
  _VOLATILE_ int idxSmem[_TPB_],
  int K
){
  int tid = threadIdx.x;
  #if _TPB_ == 32
  bitonic_sort_32(value, index, tid);

  #elif _TPB_ == 64
  bitonic_sort_64(value, index, valSmem, idxSmem, tid);
  
  #elif _TPB_ == 128
  bitonic_sort_128(value, index, valSmem, idxSmem, tid);
  
  #elif _TPB_ == 256
  bitonic_sort_256(value, index, valSmem, idxSmem, tid);
  
  #elif _TPB_ == 512
  bitonic_sort_512(value, index, valSmem, idxSmem, tid);
  
  #elif _TPB_ == 1024
  bitonic_sort_1024(value, index, valSmem, idxSmem, tid);
  
  #endif
  switch (K){
    case 2:
      bitonic_sort_global_2(
        finalValue, finalIndex,
        value, index, tid);
      break;
    case 4:
      bitonic_sort_global_4(
        finalValue, finalIndex,
        value, index, tid);
      break;
    case 8:
      bitonic_sort_global_8(
        finalValue, finalIndex,
        value, index, tid);
      break;
    case 16:
      bitonic_sort_global_16(
        finalValue, finalIndex,
        value, index, tid);
      break;
    case 32:
      bitonic_sort_global_32(
        finalValue, finalIndex,
        value, index, tid);
      break;
    case 64:
      bitonic_sort_global_64(
        finalValue, finalIndex, value, index,
        valSmem, idxSmem, tid);
      break;
    case 128:
      bitonic_sort_global_128(
        finalValue, finalIndex, value, index,
        valSmem, idxSmem, tid);
      break;
    case 256:
      bitonic_sort_global_256(
        finalValue, finalIndex, value, index,
        valSmem, idxSmem, tid);
      break;
    case 512:
      bitonic_sort_global_512(
        finalValue, finalIndex, value, index,
        valSmem, idxSmem, tid);
      break;
    case 1024:
      bitonic_sort_global_1024(
        finalValue, finalIndex, value, index,
        valSmem, idxSmem, tid);
      break;
  }
}

__device__ void load_buffer(
  const float* mat, 
  pair buffer[_TN_],
  int i, int N
){
  const ll_t iM = blockIdx.x;
  const int tid = threadIdx.x;
  #pragma unroll
  for (int j=0; j<_TN_; j++){
    ll_t iN = i * _TPB_ * _TN_ + j * _TPB_ + tid;
    if (iN < N){
      buffer[j].value = mat[iM * ll_t(N) + iN];
      buffer[j].index = iN;
    } else {
      buffer[j].value = -INFINITY;
      buffer[j].index = -1;
    }
  }
}

__device__ void arr2arr(
  pair src[_TN_],
  pair tar[_TN_]
){
  #pragma unroll
  for (int i=0; i<_TN_; i++){
    tar[i] = src[i];
  }
}

extern "C"
__global__ void topk_select(
   const float* __restrict__ mat,
   float* __restrict__ gValue,
   ll_t* __restrict__ gIndex,
   int M, int N, int K
){
  const int tid = threadIdx.x;
  const ll_t iM = blockIdx.x;
  // this is used to exchange values between threads when sorting 
  __shared__ _VOLATILE_ float valSmem[_TPB_];

  // this is used to exchange indices between threads when sorting 
  __shared__ _VOLATILE_ int idxSmem[_TPB_];

  /*
    this is used to signal that at least one threads has reached its maximum queue size,
    so that all threads will perform a bitonic sort.
  */
  __shared__ _VOLATILE_ int signal[1];
  signal[0] = 0;

  /*
    this is used to threshold the input values, values below this threashold
    will not be added to thread queue, or trigger a sort, this value is broadcasted
    from last thread to all threads at the end of each bitonic sort.
  */
  __shared__ _VOLATILE_ float minSmem[1];
  minSmem[0] = -INFINITY;

  __shared__ _VOLATILE_ pair queueSmem[_TPB_][_QCAP_];
  init_queue(queueSmem);
  __syncthreads();

  int queueFront = -1;
  int queueRear = -1;

  float minValue = -INFINITY;

  /*
    finalValue and finalIndex are the storage of final topk values and indices,
    they will be updated at each bitonic sort step, and stored to DRAM at the very end.
  */
  float finalValue = -INFINITY;
  int finalIndex = -1;
  pair buffer[_TN_];
  pair working[_TN_];
  load_buffer(mat, buffer, 0, N);

  // The number of iterations of the main loop is ceil(N / (ThreadsPerBlock * TN))
  const int nIter = (N + _TPB_ * _TN_ - 1) / (_TPB_ * _TN_);
  for (int i=0; i < nIter; i++){
    // move prefetched data from buffer to working array
    arr2arr(buffer, working);
    // then start fetching next tiles of data to buffer array
    if (i < nIter - 1){
      load_buffer(mat, buffer, i+1, N);
    }
    #pragma unroll
    for (int j=0; j < _TN_; j++){
      ll_t iN = i * _TPB_ * _TN_ + j * _TPB_ + tid;
      // prevent over-read
      if (iN >= N){
        break;
      }
      pair newPair = working[j];
      pair oldPair;
      oldPair.value = -INFINITY;
      oldPair.index = -1;

      /*
        if the queue is full, pop the front item, if the value of popped item is larger
        than previous minValue, trigger block-wise bitonic sort
      */
      if (is_queue_full(queueFront, queueRear)){
        pop_queue(queueSmem, oldPair, queueFront, queueRear);
        if (oldPair.value > minValue){
          // atomicAdd(signal, 1);
          signal[0] = 1;
        }
      }
      /*
        if incoming value is greater then previous minValue,
        add the (newValue, newIndex) pair to queue
      */
      if (newPair.value > minValue){
        push_queue(queueSmem, newPair, queueFront, queueRear);
      }
      __syncthreads();
      
      if (signal[0] > 0){
        //if any thread has triggered blockwise sort, perform sort
        sort(
          finalValue, finalIndex,
          oldPair.value, oldPair.index,
          valSmem, idxSmem, K
        );
        __syncthreads();

        // reset the signal
        signal[0] = 0;
        // last thread sets minSmem to its finalValue
        if (tid == _TPB_ - 1){
          minSmem[0] = finalValue;
        }
        __syncthreads();
        // all threads read from minSmem to set new minValue
        minValue = minSmem[0];
      }

      __syncthreads();
    }
  }
  // pop all remaining items from queue
  for (int i=0; i<_QCAP_; i++){
    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (!is_queue_empty(queueFront, queueRear)){
      pop_queue(queueSmem, oldPair, queueFront, queueRear);
      if (oldPair.value > minValue){
        //atomicAdd(signal, 1);
        signal[0] = 1;
      }
    }
    __syncthreads();
    if (signal[0] > 0){
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        valSmem, idxSmem, K
      );
      __syncthreads();

      signal[0] = 0;
      if (tid == _TPB_ - 1){
        minSmem[0] = finalValue;
      }
      __syncthreads();
      minValue = minSmem[0];
    }

    __syncthreads();
  }
  // last K threads write their finalValue and finalIndex to gValue and gIndex
  if (_TPB_ - K <= tid){
    const int writeAddress = (iM * K) + tid - ( _TPB_ - K);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = ll_t(finalIndex);
  }
}