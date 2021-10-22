#define _VOLATILE_ 
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define isnan(x) ( x != x )
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#define N_WARPS _TPB_/32


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

__device__ __inline__ bool is_queue_full_smem(
  int queueFront,
  int queueRear
){
  return ((queueFront - queueRear) == 1 || (queueFront == 0 && queueRear == _QCAP_ - 1));
  //return (queueRear + 1) % _QCAP_ == queueFront;
}

__device__ __inline__ bool is_queue_empty_smem(
  int queueFront,
  int queueRear
){
  return queueFront == -1;
}

__device__ void push_queue_smem(
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

__device__ void pop_queue_smem(
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

__device__ void push_pop_queue_smem(
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

__device__ void init_queue_smem(
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


// For k == 32
extern "C"
__global__ void topk_select_v2(
  const float* __restrict__ mat,
  float* gValue,
  ll_t* gIndex,
  int M, int N
){
  const int tid = threadIdx.x; // thread id within block
  const int wx = tid % 32; // thread id within warp
  const int wy = tid / 32; // warp id within block
  const int mStart = blockIdx.x * N_WARPS;
  const int mid = mStart + wy; // each warp is responsible for a row (or column?) of matrix

  __shared__ _VOLATILE_ float valSmem[N_WARPS][32];
  __shared__ _VOLATILE_ int idxSmem[N_WARPS][32];
  int signal = 0;
  float minValue = -INFINITY;
  float finalValue = -INFINITY;
  int finalIndex = -1;

  __shared__ _VOLATILE_ pair queueSmem[N_WARPS][32][_QCAP_];
  init_queue_smem(queueSmem);
  __syncwarp();
  int queueFront = -1;
  int queueRear = -1;
  pair working[_TN_];

  const int nIter = (n + 32 - 1) / 32;
  for (int i = 0; i < nIter; i++){
    load_buffer(mat, working, i, mid, N);
  }
  #pragma unroll
  for (int j=0; j < _TN_; j++){
    pair newPair = working[j];
    pair oldPair
    oldPair.value = -INFINITY;
    oldPair.index = -1;
    if (is_queue_full(queueFront, queueRear)){
      pop_queue(queueSmem, oldPair, queueFront, queueRear);
      if (oldPair.value > minValue){
        signal = 1;
      }
    }

    if (newPair.value > minValue){
      push_queue(queueSmem, newPair, queueFront, queueRear);
    }
    __syncwarp();

    if (signal[0] > 0){}
  }
}