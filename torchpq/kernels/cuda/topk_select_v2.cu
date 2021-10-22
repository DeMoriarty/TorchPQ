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

__device__ __forceinline__ void load_buffer(
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
__global__ void topk_select_v2(
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

  // this is used to exchange values between threads when sorting 
  // __shared__ _VOLATILE_ float valSmem[N_WARPS][32];

  // this is used to exchange indices between threads when sorting 
  // __shared__ _VOLATILE_ int idxSmem[N_WARPS][32];

  /*
    this is used to signal that at least one threads has reached its maximum queue size,
    so that all threads will perform a bitonic sort.
  */
  __shared__ _VOLATILE_ int signal[N_WARPS];
  if (wx == 0){
    signal[wy] = 0;
  }

  /*
    this is used to threshold the input values, values below this threashold
    will not be added to thread queue, or trigger a sort, this value is broadcasted
    from last thread to all threads at the end of each bitonic sort.
  */
  // __shared__ _VOLATILE_ float minSmem[N_WARPS];
  // if (wx == 0){
  //   minSmem[wy] = -INFINITY;
  // }

  __shared__ _VOLATILE_ pair queueSmem[N_WARPS][32][_QCAP_];
  init_queue(queueSmem, wx, wy);
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
  load_buffer(mat, buffer, 0, iM, wx, N);

  // The number of iterations of the main loop is ceil(N / (ThreadsPerBlock * TN))
  const int nIter = (N + 32 * _TN_ - 1) / (32 * _TN_);
  for (int i=0; i < nIter; i++){
    // move prefetched data from buffer to working array
    arr2arr(buffer, working);
    // then start fetching next tiles of data to buffer array
    if (i < nIter - 1){
      load_buffer(mat, buffer, i+1, iM, wx, N);
    }
    #pragma unroll
    for (int j=0; j < _TN_; j++){
      pair newPair = working[j];
      pair oldPair;
      oldPair.value = -INFINITY;
      oldPair.index = -1;

      /*
        if the queue is full, pop the front item, if the value of popped item is larger
        than previous minValue, trigger block-wise bitonic sort
      */
      if (is_queue_full(queueFront, queueRear)){
        pop_queue(queueSmem, oldPair, queueFront, queueRear, wx, wy);
        if (oldPair.value > minValue){
          // atomicAdd(signal, 1);
          signal[wy] = 1;
        }
      }
      /*
        if incoming value is greater then previous minValue,
        add the (newValue, newIndex) pair to queue
      */
      if (newPair.value > minValue){
        push_queue(queueSmem, newPair, queueFront, queueRear, wx, wy);
      }
      __syncwarp();
      
      if (signal[wy] > 0){
        //if any thread has triggered blockwise sort, perform sort
        sort(
          finalValue, finalIndex,
          oldPair.value, oldPair.index,
          K
        );
        __syncwarp();

        // reset the signal
        signal[wy] = 0;
        minValue = __shfl_sync(0xffffffff, finalValue, 31);
      }

      __syncwarp();
    }
  }
  // pop all remaining items from queue
  for (int i=0; i<_QCAP_; i++){
    pair oldPair;
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (!is_queue_empty(queueFront, queueRear)){
      pop_queue(queueSmem, oldPair, queueFront, queueRear, wx, wy);
      if (oldPair.value > minValue){
        //atomicAdd(signal, 1);
        signal[wy] = 1;
      }
    }
    __syncwarp();
    if (signal[wy] > 0){
      sort(
        finalValue, finalIndex,
        oldPair.value, oldPair.index,
        K
      );
      __syncwarp();

      signal[wy] = 0;
      minValue = __shfl_sync(0xffffffff, finalValue, 31);
    }

    __syncwarp();
  }
  // last K threads write their finalValue and finalIndex to gValue and gIndex
  if (32 - K <= wx){
    const int writeAddress = (iM * K) + wx - (32 - K);
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = ll_t(finalIndex);
  }
}