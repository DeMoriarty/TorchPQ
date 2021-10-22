#define _VOLATILE_ 
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define N_WARPS _TPB_/32
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#define get_cell_info(arr, x, y, z) arr[x * maxNProbe * 3 + y * 3 + z]

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
  unsigned int &cell,
  unsigned int &index,
  const int stride
){
  const float otherValue = __shfl_down_sync(0xFFFFFFFF, value, stride, 32);
  const unsigned int otherIndex = __shfl_down_sync(0xFFFFFFFF, index, stride, 32);
  const unsigned int otherCell = __shfl_down_sync(0xFFFFFFFF, cell, stride, 32);
  bool condition = otherValue > value;
  value = condition ? otherValue : value;
  index = condition ? otherIndex : index;
  cell = condition ? otherCell : cell;
}

__device__ __forceinline__ void thread_comparator(
  float &value,
  unsigned int &cell,
  unsigned int &index,
  float otherValue,
  unsigned int otherCell,
  unsigned int otherIndex
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
    cell = otherCell;
  }
}


__device__ __forceinline__ void max_2(
  float &value,
  unsigned int &cell,
  unsigned int &index
){
  warp_comparator(value, cell, index, 1);
}

__device__ __forceinline__ void max_4(
  float &value,
  unsigned int &cell,
  unsigned int &index
){
  warp_comparator(value, cell, index, 2);
  max_2(value, cell, index);
}

__device__ __forceinline__ void max_8(
  float &value,
  unsigned int &cell,
  unsigned int &index
){
  warp_comparator(value, cell, index, 4);
  max_4(value, cell, index);
}

__device__ __forceinline__ void max_16(
  float &value,
  unsigned int &cell,
  unsigned int &index
){
  warp_comparator(value, cell, index, 8);
  max_8(value, cell, index);
}

__device__ __forceinline__ void max_32(
  float &value,
  unsigned int &cell,
  unsigned int &index
){
  warp_comparator(value, cell, index, 16);
  max_16(value, cell, index);
}

__device__ void block_max(
  float &value,
  unsigned int &cell,
  unsigned int &index,
  _VOLATILE_ float sMem[]
){
  max_32(value, cell, index);
  #if N_WARPS > 1
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  float temp1;
  unsigned int temp2, temp3;
  float otherValue;
  unsigned int otherIndex, otherCell;
  _VOLATILE_ unsigned int* uintSMem = (unsigned int*) sMem;
  if (wx == 0){
    temp1 = sMem[wy];
    temp2 = uintSMem[N_WARPS + wy];
    temp3 = uintSMem[N_WARPS * 2 + wy];
  }
  __syncthreads();

  if (wx == 0){
    sMem[wy] = value;
    uintSMem[N_WARPS + wy] = cell;
    uintSMem[N_WARPS * 2 + wy] = index;
  }
  __syncthreads();

  if (tid < N_WARPS){
    value = sMem[tid];
    cell = uintSMem[N_WARPS + tid];
    index = uintSMem[N_WARPS * 2 + tid];
  }
  __syncthreads();

  if (wx == 0){
    sMem[wy] = temp1;
    uintSMem[N_WARPS + wy] = temp2;
    uintSMem[N_WARPS * 2 + wy] = temp3;
  }
  __syncthreads();
  if (wy == 0){
    #if N_WARPS == 2
    max_2(value, cell, index);

    #elif N_WARPS == 4
    max_4(value, cell, index);

    #elif N_WARPS == 8
    max_8(value, cell, index);

    #elif N_WARPS == 16
    max_16(value, cell, index);

    #elif N_WARPS == 32
    max_32(value, cell, index);

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

// FIXME:
__device__ void load_consume_data(
  const uint8n_t* cellPtr,
  _VOLATILE_ float sMem[],
  float &value,
  unsigned int itemIndex,
  unsigned int cellCapacity
){
  #pragma unroll
  for (int i = 0; i < _M_ / _NCS_; i++){
    uint8n_t threadData = cellPtr[(i * cellCapacity) + itemIndex];
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

// FIXME:
__device__ void load_data(
  const uint8n_t* cellPtr,
  uint8n_t dataCache[_M_ / _NCS_],
  unsigned int itemIndex,
  unsigned int cellCapacity
){
  #pragma unroll
  for (int i = 0; i < _M_ / _NCS_; i++){
    uint8n_t threadData = cellPtr[(i * cellCapacity) + itemIndex];
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
  // const uint8n_t* __restrict__ data,
  const ll_t* __restrict__ addressToIdPtr, //[nCell]
  const float* __restrict__ precomputed,
  const ll_t* __restrict__ cellInfo, //[nQuery, maxNProbe, 3]
  const ll_t* __restrict__ totalSize, //[nQuery]
  const ll_t* __restrict__ nProbeList, //[nQuery]
  float* __restrict__ gValue,
  ll_t* __restrict__ gAddress,
  ll_t* __restrict__ gID,
  int nQuery, int maxNProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  const int nProbe = nProbeList[qid];

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  load_precomputed(precomputed, sMem, nQuery);
  float finalValue = -INFINITY;
  unsigned int finalIndex = -1;
  unsigned int finalCell = -1;
  const ll_t threadTotalSize = totalSize[qid];
  const int nIter = (threadTotalSize + _TPB_ - 1) / _TPB_;
  unsigned int cCell = 0;
  // int cCellStart = cellStart[qid * maxNProbe + cCell];
  // int cCellSize = cellSize[qid * maxNProbe + cCell];
  const uint8n_t* cCellPtr = (const uint8n_t*) get_cell_info(cellInfo, qid, cCell, 1);
  unsigned int cCellSize = get_cell_info(cellInfo, qid, cCell, 0);
  unsigned int cCellCapacity = get_cell_info(cellInfo, qid, cCell, 2);
  // int cCellEnd = cCellStart + cCellSize;
  // int iN = cCellStart + tid;
  unsigned int cItemIndex = tid;

  for (int i = 0; i < nIter; i++){
    while (cItemIndex >= cCellSize){
      cCell ++;  // increment cell index by 1
      if (cCell >= nProbe)
        break;
      const uint8n_t* pCellPtr = cCellPtr;
      unsigned int pCellSize = cCellSize;

      const uint8n_t* cCellPtr = (const uint8n_t*) get_cell_info(cellInfo, qid, cCell, 1);
      if (cCellPtr == pCellPtr){
        continue;
      }
      cCellSize = get_cell_info(cellInfo, qid, cCell, 0);
      cCellCapacity = get_cell_info(cellInfo, qid, cCell, 2);
      cItemIndex = cItemIndex - pCellSize + 0 // FIXME:
    }
    float value;
    // float index = iN;
    if (cItemIndex < cCellSize){
      value = 0.f;
      uint8n_t dataCache[_M_ / _NCS_];
      load_data(cCellPtr, dataCache, cItemIndex, cCellCapacity);
      consume_data(sMem, dataCache, value); 
    } else {
      value = -INFINITY;
    }

    thread_comparator(finalValue, finalCell, finalIndex, value, cCell, cItemIndex);
    cItemIndex += _TPB_;
  }
  block_max(finalValue, finalCell, finalIndex, sMem);

  if (tid == 0){
    gValue[qid] = finalValue;
    gAddress[qid * 2 + 0] = finalCell;
    gAddress[qid * 2 + 1] = finalIndex;
  }
}

extern "C"
__global__ void ivfpq_top1_residual(
  // const uint8n_t* __restrict__ data,
  const ll_t* __restrict__ addressToIdPtr,
  const float* __restrict__ precomputed,
  const float* __restrict__ baseSims,
  const ll_t* __restrict__ cellInfo, //[nQuery, maxNProbe, 3]
  const ll_t* __restrict__ totalSize,
  const ll_t* __restrict__ nProbeList,
  float* __restrict__ gValue,
  ll_t* __restrict__ gAddress,
  ll_t* __restrict__ gID,
  int nQuery, int maxNProbe
) {
  // const int tid = threadIdx.x; // thread ID
  // const int qid = blockIdx.x; // query ID
  // const int nProbe = nProbeList[qid];

  // extern __shared__ _VOLATILE_ float sMem[]; // M * K
  // const ll_t threadTotalSize = totalSize[qid];
  // float finalValue = -654321;
  // float finalIndex = -1;

  // int cCellStart = -1;
  // for (int cCell = 0; cCell < nProbe; cCell++){
  //   int pCellStart = cCellStart;
  //   cCellStart = cellStart[qid * maxNProbe + cCell];
  //   if (cCellStart == pCellStart){
  //     continue;
  //   }
  //   int cCellSize = cellSize[qid * maxNProbe + cCell];
  //   load_precomputed_v2(precomputed, sMem, cCell, maxNProbe);
  //   float cBaseSim = baseSims[qid * maxNProbe + cCell];
  //   int cCellEnd = cCellStart + cCellSize;
  //   int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
  //   for (int iter = 0; iter < nIter; iter++ ){
  //     int iN = cCellStart + iter * _TPB_ + tid;
  //     float value;
  //     float index = iN;
  //     int cIsEmpty = 0;
  //     if (cCellStart <= iN && iN < cCellEnd){
  //       value = cBaseSim;
  //       cIsEmpty = isEmpty[iN];
  //       uint8n_t dataCache[_M_ / _NCS_];
  //       load_data(data, dataCache, iN, nData);
  //       consume_data(sMem, dataCache, value);
  //     } else {
  //       value = -123456.f;
  //     }
  //     value = cIsEmpty == 0 ? value : -987654.f;
  //     index = cIsEmpty == 0 ? index : -1;
  //     thread_comparator(finalValue, finalIndex, value, index);
  //   }
  // }
  // block_max(finalValue, finalIndex, sMem);

  // if (tid == 0){
  //   const int writeAddress = qid;
  //   gValue[writeAddress] = finalValue;
  //   gIndex[writeAddress] = finalIndex;
  // }
}

extern "C"
__global__ void ivfpq_top1_residual_precomputed(
  // const uint8n_t* __restrict__ data,
  const ll_t* __restrict__ addressToIdPtr,
  const float* __restrict__ part1,
  const float* __restrict__ part2,
  const ll_t* __restrict__ cells,
  const float* __restrict__ baseSims,
  const ll_t* __restrict__ cellInfo, // [nQuery, maxNProbe, 3]
  const ll_t* __restrict__ totalSize,
  const ll_t* __restrict__ nProbeList,
  float* __restrict__ gValue,
  ll_t* __restrict__ gAddress,
  ll_t* __restrict__ gID,
  int nQuery, int maxNProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  const int nProbe = nProbeList[qid];

  extern __shared__ _VOLATILE_ float sMem[]; // M * K
  const ll_t threadTotalSize = totalSize[qid];
  float finalValue = -INFINITY;
  unsigned int finalIndex = -1;
  unsigned int finalCell = -1;
  float part1Cache[_M_];
  float part2Cache[_M_];
  load_part1_to_cache(part1, part1Cache);

  // int nCellStart = cellStart[qid * maxNProbe];
  // int nCellSize = cellSize[qid * maxNProbe];
  // int nCellEnd = nCellStart + nCellSize;
  unsigned nCellPtr
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
      float value;
      float index = iN;
      int cIsEmpty = 0;
      if (iN < cCellEnd){
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