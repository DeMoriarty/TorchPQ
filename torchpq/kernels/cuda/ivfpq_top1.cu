#define _VOLATILE_ 
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define N_WARPS _TPB_/32
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

typedef unsigned char uint8_t;
typedef long long ll_t;

typedef struct __device_builtin__ __builtin_align__(_NCS_)
{
  uint8_t _VARNAMES_;
} _uint8n_t;

typedef union {
  _uint8n_t u8n;
  uint8_t val[_NCS_];
} uint8n_t;

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
  const int stride
){
  const float otherValue = __shfl_down_sync(0xFFFFFFFF, value, stride, 32);
  const float otherIndex = __shfl_down_sync(0xFFFFFFFF, index, stride, 32);
  bool condition = otherValue > value;
  value = condition ? otherValue : value;
  index = condition ? otherIndex : index;
}

__device__ __forceinline__ void thread_comparator(
  float &value,
  float &index,
  float otherValue,
  float otherIndex
){
  bool condition = otherValue > value;
  if (condition){
    /*
    value = value + otherValue;
    otherValue = value - otherValue;
    value = value - otherValue;
    index = index + otherIndex;
    otherIndex = index - otherIndex;
    index = index - otherIndex;
    */
    value = otherValue;
    index = otherIndex;
  }
}


__device__ __forceinline__ void max_2(
  float &value,
  float &index
){
  warp_comparator(value, index, 1);
}

__device__ __forceinline__ void max_4(
  float &value,
  float &index
){
  warp_comparator(value, index, 2);
  max_2(value, index);
}

__device__ __forceinline__ void max_8(
  float &value,
  float &index
){
  warp_comparator(value, index, 4);
  max_4(value, index);
}

__device__ __forceinline__ void max_16(
  float &value,
  float &index
){
  warp_comparator(value, index, 8);
  max_8(value, index);
}

__device__ __forceinline__ void max_32(
  float &value,
  float &index
){
  warp_comparator(value, index, 16);
  max_16(value, index);
}

__device__ void block_max(
  float &value,
  float &index,
  _VOLATILE_ float sMem[]
){
  max_32(value, index);
  #if N_WARPS > 1
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  float temp1, temp2;
  float otherValue, otherIndex;
  if (wx == 0){
    temp1 = sMem[wy];
    temp2 = sMem[N_WARPS + wy];
  }
  __syncthreads();

  if (wx == 0){
    sMem[wy] = value;
    sMem[N_WARPS + wy] = index;
  }
  __syncthreads();

  if (tid < N_WARPS){
    value = sMem[tid];
    index = sMem[N_WARPS + tid];
  }
  __syncthreads();

  if (wx == 0){
    sMem[wy] = temp1;
    sMem[N_WARPS + wy] = temp2;
  }
  __syncthreads();
  if (wy == 0){
    #if N_WARPS == 2
    max_2(value, index);

    #elif N_WARPS == 4
    max_4(value, index);

    #elif N_WARPS == 8
    max_8(value, index);

    #elif N_WARPS == 16
    max_16(value, index);

    #elif N_WARPS == 32
    max_32(value, index);

    #endif
  }
  #endif
}

__device__ void load_precomputed(
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
      precomputedValue += part2[adr2];
      sMem[i * _K_ + tid] = precomputedValue;

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
        precomputedValue += part2[adr2];
        sMem[i * _K_ + j * _TPB_ + tid] = precomputedValue;
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

extern "C"
__global__ void ivfpq_top1(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int nProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  load_precomputed(precomputed, sMem, nQuery);
  float finalValue = -INFINITY;
  float finalIndex = -1;
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
      if (cCell >= nProbe)
        break;
      int pCellEnd = cCellEnd;
      cCellStart = cellStart[qid * nProbe + cCell];
      cCellSize = cellSize[qid * nProbe + cCell];
      cCellEnd = cCellStart + cCellSize;
      iN = iN - pCellEnd + cCellStart;
    }
    float value;
    float index = iN;
    int cIsEmpty = 0;
    if (cCellStart <= iN && iN < cCellEnd){
      value = 0.f;
      cIsEmpty = isEmpty[iN];

      uint8n_t dataCache[_M_ / _NCS_];
      load_data(data, dataCache, iN, nData);
      consume_data(sMem, dataCache, value);
    } else {
      value = -INFINITY;
    }
    value = cIsEmpty == 0 ? value : -INFINITY;
    index = cIsEmpty == 0 ? index : -1;

    thread_comparator(finalValue, finalIndex, value, index);
    iN += _TPB_;
  }
  block_max(finalValue, finalIndex, sMem);

  if (tid == 0){
    const int writeAddress = qid;
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_top1_residual(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const float* __restrict__ baseSims,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int nProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -654321;
  float finalIndex = -1;

  for (int cCell = 0; cCell < nProbe; cCell++){
    int cCellStart = cellStart[qid * nProbe + cCell];
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
        value = -123456.f;
      }
      value = cIsEmpty == 0 ? value : -987654.f;
      index = cIsEmpty == 0 ? index : -1;
      thread_comparator(finalValue, finalIndex, value, index);
    }
  }
  block_max(finalValue, finalIndex, sMem);

  if (tid == 0){
    const int writeAddress = qid;
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_top1_residual_precomputed_v1(
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
  int nData, int nQuery, int nProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -654321;
  float finalIndex = -1;

  for (int cCell = 0; cCell < nProbe; cCell++){
    int cCellStart = cellStart[qid * nProbe + cCell];
    int cCellSize = cellSize[qid * nProbe + cCell];
    int iCell = cells[qid * nProbe + cCell];
    load_precomputed_v3(part1, part2, sMem, iCell);
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
        value = -123456.f;
      }
      value = cIsEmpty == 0 ? value : -987654.f;
      index = cIsEmpty == 0 ? index : -1;
      thread_comparator(finalValue, finalIndex, value, index);
    }
  }
  block_max(finalValue, finalIndex, sMem);

  if (tid == 0){
    const int writeAddress = qid;
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_top1_residual_precomputed(
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
  int nData, int nQuery, int nProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -INFINITY;
  float finalIndex = -1;
  float part1Cache[_M_];
  float part2Cache[_M_];
  load_part1_to_cache(part1, part1Cache);

  int nCellStart = cellStart[qid * nProbe];
  int nCellSize = cellSize[qid * nProbe];
  int nCellEnd = nCellStart + nCellSize;
  int iCell = cells[qid * nProbe];
  load_part2_to_cache(part2, part2Cache, iCell);

  for (int cCell = 0; cCell < nProbe; cCell++){
    int cCellStart = nCellStart;
    int cCellSize = nCellSize;
    int cCellEnd = nCellEnd;
    store_precomputed_to_smem(part1Cache, part2Cache, sMem);

    if (cCell < nProbe - 1){
      nCellStart = cellStart[qid * nProbe + cCell + 1];
      nCellSize = cellSize[qid * nProbe + cCell + 1];
      nCellEnd = nCellStart + nCellSize;
      iCell = cells[qid * nProbe + cCell + 1];
      load_part2_to_cache(part2, part2Cache, iCell);
    }

    float cBaseSim = baseSims[qid * nProbe + cCell];
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
      thread_comparator(finalValue, finalIndex, value, index);
    }
  }
  block_max(finalValue, finalIndex, sMem);

  if (tid == 0){
    const int writeAddress = qid;
    gValue[writeAddress] = finalValue;
    gIndex[writeAddress] = finalIndex;
  }
}