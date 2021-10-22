#define _VOLATILE_ 
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)
#define __prefetch_l1(adr) asm("prefetch.L1 [%0];" :: "l"(adr) )
#define __prefetch_l2(adr) asm("prefetch.L2 [%0];" :: "l"(adr) )
#define get_precomputed_from_smem(arr, x, y, z) arr[(x) * _M_ * _K_ + (y) * _K_ + (z)]
#define get_precomputed_from_regf(val, x) __shfl_sync(0xffffffff, val, x);

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#define N_WARPS _TPB_/32
#define HALFM _M_ / 2

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


__device__ __forceinline__ void block_max(
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

__device__ __forceinline__ void prefetch_precomputed(
  const float *precomputed, // [M, nQuery, K]
  int qid,
  int nQuery
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int hwx = wx % 16;
  const int hwy = wx / 16;
  #pragma unroll
  for (int i = 0; i < HALFM; i++){
    if (i + HALFM * hwy < _M_){
      const float* adr = precomputed + ((i + HALFM * hwy) * nQuery * _K_) + (qid * _K_) + (hwx);
      __prefetch_l1(adr);
    }
  }
}

__device__ __forceinline__ void load_precomputed_to_smem(
  const float *precomputed, // [M, nQuery, K]
  _VOLATILE_ float *precomputedSMEM, // [N_WARP, M, K]
  int qid,
  int nQuery
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  const int hwx = wx % 16;
  const int hwy = wx / 16;
  #pragma unroll
  for (int i = 0; i < HALFM; i++){
    if (i + HALFM * hwy < _M_){
      const int adr = ((i + HALFM * hwy) * nQuery * _K_) + (qid * _K_) + (hwx);
      precomputedSMEM[
        (wy) * _M_ * _K_ + 
        (i +  HALFM * hwy) * _K_ 
        + (hwx)
      ] = precomputed[adr];
    }
  }
}

__device__ __forceinline__ void load_precomputed_to_regf(
  const float *precomputed, // [M, nQuery, K]
  float precomputedREGF[HALFM],
  int qid,
  int nQuery
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  const int hwx = wx % 16;
  const int hwy = wx / 16;
  #pragma unroll
  for (int i = 0; i < HALFM; i++){
    if (i + HALFM * hwy < _M_){
      const int adr = ((i + HALFM * hwy) * nQuery * _K_) + (qid * _K_) + (hwx);
      precomputedREGF[i] = precomputed[adr];
    }
  }
}

__device__ __forceinline__ void load_precomputed_v2(
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

__device__ __forceinline__ void load_precomputed_v3(
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

__device__ __forceinline__ void load_part1_to_cache(
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

__device__ __forceinline__ void load_part2_to_cache(
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

__device__ __forceinline__ void store_precomputed_to_smem(
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

__device__ __forceinline__ void load_consume_data(
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

__device__ __forceinline__ void load_data(
  const uint8n_t* data, //[HALFM / NCS, nData, NCS]
  uint8_t dataCache[HALFM],
  int iN, int nData
){
  #pragma unroll
  for (int i = 0; i < HALFM / _NCS_; i++){
    uint8n_t threadData = data[(i * nData) + iN];
    #pragma unroll
    for (int j = 0; j < _NCS_; j++){
      dataCache[i * _NCS_ + j] = threadData.val[j];
    }
  }
}

__device__ __forceinline__ void prefetch_data(
  const uint8n_t* data,
  int iN, int nData
){
  #pragma unroll
  for (int i = 0; i < HALFM / _NCS_; i++){
    const uint8n_t* adr = data + (i * nData) + iN;
    __prefetch_l2(adr);
  }
}

__device__ __forceinline__ void consume_data_smem(
  _VOLATILE_ float precomputedSMEM[], //[N_WARP, M, K]
  uint8_t dataCache[HALFM],
  float &value
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  const int hwx = wx % 16;
  const int hwy = wx / 16;
  #pragma unroll
  for (int i = 0; i < HALFM; i++){
    uint8_t threadData = dataCache[i];
    // int first = threadData >> 4;
    // int second = threadData & 0x0f;
    value += get_precomputed_from_smem(precomputedSMEM, wy, i, threadData);
    // value += get_precomputed_from_smem(precomputedSMEM, wy, i + HALFM, second);
  }
}

__device__ __forceinline__ void consume_data_regf(
  // _VOLATILE_ float precomputedSMEM[], //[N_WARP, M, K]
  float precomputedREGF[HALFM],
  uint8_t dataCache[HALFM],
  float &value
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  const int hwx = wx % 16;
  const int hwy = wx / 16;
  #pragma unroll
  for (int i = 0; i < HALFM; i++){
    uint8_t threadData = dataCache[i];
    int first = threadData >> 4;
    int second = threadData & 0x0f;
    value += get_precomputed_from_regf(precomputedREGF[i], first);
    value += get_precomputed_from_regf(precomputedREGF[i], 16 + second);
    // value += get_precomputed_from_smem(precomputedSMEM, wy, i, threadData);
    // value += get_precomputed_from_smem(precomputedSMEM, wy, i + HALFM, second);
  }
}


__device__ __forceinline__ void consume_data_smem_v1(
  _VOLATILE_ float precomputedSMEM[], //[N_WARP, M, K]
  uint8n_t dataCache[HALFM / _NCS_],
  float &value
){
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  const int hwx = wx % 16;
  const int hwy = wx / 16;
  #pragma unroll
  for (int i = 0; i < HALFM / _NCS_; i++){
    uint8n_t threadData = dataCache[i];
    #pragma unroll
    for (int j = 0; j < _NCS_; j++){
      int idx = threadData.val[j] >> 4;
      value += idx;
      // float pre = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + j, idx );
      // value += pre;
    }
    #pragma unroll
    for (int j = 0; j < _NCS_; j++){    
      int idx = threadData.val[j] & 0x0f;
      value += idx;
      // float pre = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + j, idx );
      // value += pre;
    }
    // float pre0 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 0, threadData.val[0] >> 4 );
    // float pre1 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 1, threadData.val[1] >> 4 );
    // float pre2 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 2, threadData.val[2] >> 4 );
    // float pre3 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 3, threadData.val[3] >> 4 );
    // float pre4 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 0, threadData.val[0] & 0x0f );
    // float pre5 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 1, threadData.val[1] & 0x0f );
    // float pre6 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 2, threadData.val[2] & 0x0f );
    // float pre7 = get_precomputed_from_smem(precomputedSMEM, wy, i * _NCS_ + 3, threadData.val[3] & 0x0f );
    // value += pre0;
    // value += pre1;
    // value += pre2;
    // value += pre3;
    // value += pre4;
    // value += pre5;
    // value += pre6;
    // value += pre7;
  }
}

extern "C"
__global__ void ivfpq4_top1_v1(
  const uint8n_t* __restrict__ data,
  const float* __restrict__ precomputed,
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ cellStart,
  const ll_t* __restrict__ cellSize,
  const ll_t* __restrict__ totalSize,
  const ll_t* __restrict__ nProbeList,
  float* __restrict__ gValue,
  ll_t* __restrict__ gIndex,
  int nData, int nQuery, int maxNProbe
) {
  const int tid = threadIdx.x;
  const int wx = tid % 32;
  const int wy = tid / 32;
  const int qStart = blockIdx.x * N_WARPS;
  const int qid = tid + wy;
  if (qid >= nQuery){
    return;
  }
  const int nProbe = nProbeList[qid];
  const ll_t threadTotalSize = totalSize[qid];

  // extern __shared__ _VOLATILE_ float precomputedSMEM[]; // [N_WARPS, M, K]
  prefetch_precomputed(precomputed, qid, nQuery);
  float finalValue = -INFINITY;
  float finalIndex = -1;
  int nIter = (threadTotalSize + 31) / 32;
  int cCell = 0;
  int cCellStart = cellStart[qid * maxNProbe + cCell];
  int cCellSize = cellSize[qid * maxNProbe + cCell];
  int cCellEnd = cCellStart + cCellSize;
  int iN = cCellStart + tid;
  int prevIN, prevCellStart, prevCellEnd;
  float precomputedREGF[HALFM];
  // load_precomputed_to_regf(precomputed, precomputedSMEM, qid, nQuery);
  load_precomputed_to_regf(precomputed, precomputedREGF, qid, nQuery);

  for (int i = 0; i < nIter + 1; i++){
    if (i < nIter){
      while (iN >= cCellEnd){
        cCell ++;  // increment cell index by 1
        if (cCell >= nProbe)
          break;
        int pCellStart = cCellStart;
        int pCellEnd = cCellEnd;
        cCellStart = cellStart[qid * maxNProbe + cCell];
        if (cCellStart == pCellStart){
          continue;
        }
        cCellSize = cellSize[qid * maxNProbe + cCell];
        cCellEnd = cCellStart + cCellSize;
        iN = iN - pCellEnd + cCellStart;
      }
      if (iN < cCellEnd){
        // prefetch_data(data, iN, nData);
        // __prefetch_l2(isEmpty + iN);
      }
    }
    if (i > 0){
      float value;
      float index = prevIN;
      int cIsEmpty = 0;
      if (prevIN < prevCellEnd){
        value = 0.f;
        cIsEmpty = isEmpty[prevIN];
        uint8_t dataCache[HALFM];
        load_data(data, dataCache, prevIN, nData);
        // consume_data_smem(precomputedSMEM, dataCache, value);
        consume_data_regf(precomputedREGF, dataCache, value);
      } else {
        value = -INFINITY;
      }
      value = cIsEmpty == 0 ? value : -INFINITY;
      index = cIsEmpty == 0 ? index : -1;
      thread_comparator(finalValue, finalIndex, value, index);
    }
    prevCellStart = cCellStart;
    prevCellEnd = cCellEnd;
    prevIN = iN;
    iN += 32;
  }
  max_32(finalValue, finalIndex);

  if (wx == 0){
    gValue[qid] = finalValue;
    gIndex[qid] = finalIndex;
  }
}

// extern "C"
// __global__ void ivfpq_top1_residual_v2(
//   const uint8n_t* __restrict__ data,
//   const float* __restrict__ precomputed,
//   const float* __restrict__ baseSims,
//   const uint8_t* __restrict__ isEmpty,
//   const ll_t* __restrict__ cellStart,
//   const ll_t* __restrict__ cellSize,
//   const ll_t* __restrict__ totalSize,
//   const ll_t* __restrict__ nProbeList,
//   float* __restrict__ gValue,
//   ll_t* __restrict__ gIndex,
//   int nData, int nQuery, int maxNProbe
// ) {
//   const int tid = threadIdx.x; // thread ID
//   const int qid = blockIdx.x; // query ID
//   const int nProbe = nProbeList[qid];

//   extern __shared__ _VOLATILE_ float sMem[]; // M * K
//   const ll_t threadTotalSize = totalSize[qid];
//   float finalValue = -654321;
//   float finalIndex = -1;

//   int cCellStart = -1;
//   for (int cCell = 0; cCell < nProbe; cCell++){
//     int pCellStart = cCellStart;
//     cCellStart = cellStart[qid * maxNProbe + cCell];
//     if (cCellStart == pCellStart){
//       continue;
//     }
//     int cCellSize = cellSize[qid * maxNProbe + cCell];
//     load_precomputed_v2(precomputed, sMem, cCell, maxNProbe);
//     float cBaseSim = baseSims[qid * maxNProbe + cCell];
//     int cCellEnd = cCellStart + cCellSize;
//     int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
//     for (int iter = 0; iter < nIter; iter++ ){
//       int iN = cCellStart + iter * _TPB_ + tid;
//       float value;
//       float index = iN;
//       int cIsEmpty = 0;
//       if (cCellStart <= iN && iN < cCellEnd){
//         value = cBaseSim;
//         cIsEmpty = isEmpty[iN];
//         uint8n_t dataCache[_M_ / _NCS_];
//         load_data(data, dataCache, iN, nData);
//         consume_data(sMem, dataCache, value);
//       } else {
//         value = -123456.f;
//       }
//       value = cIsEmpty == 0 ? value : -987654.f;
//       index = cIsEmpty == 0 ? index : -1;
//       thread_comparator(finalValue, finalIndex, value, index);
//     }
//   }
//   block_max(finalValue, finalIndex, sMem);

//   if (tid == 0){
//     const int writeAddress = qid;
//     gValue[writeAddress] = finalValue;
//     gIndex[writeAddress] = finalIndex;
//   }
// }

// extern "C"
// __global__ void ivfpq_top1_residual_precomputed_v2(
//   const uint8n_t* __restrict__ data,
//   const float* __restrict__ part1,
//   const float* __restrict__ part2,
//   const ll_t* __restrict__ cells,
//   const float* __restrict__ baseSims,
//   const uint8_t* __restrict__ isEmpty,
//   const ll_t* __restrict__ cellStart,
//   const ll_t* __restrict__ cellSize,
//   const ll_t* __restrict__ totalSize,
//   const ll_t* __restrict__ nProbeList,
//   float* __restrict__ gValue,
//   ll_t* __restrict__ gIndex,
//   int nData, int nQuery, int maxNProbe
// ) {
//   const int tid = threadIdx.x; // thread ID
//   const int qid = blockIdx.x; // query ID
//   const int nProbe = nProbeList[qid];

//   extern __shared__ _VOLATILE_ float sMem[]; // M * K
//   const ll_t threadTotalSize = totalSize[qid];
//   float finalValue = -INFINITY;
//   float finalIndex = -1;
//   float part1Cache[_M_];
//   float part2Cache[_M_];
//   load_part1_to_cache(part1, part1Cache);

//   int nCellStart = cellStart[qid * maxNProbe];
//   int nCellSize = cellSize[qid * maxNProbe];
//   int nCellEnd = nCellStart + nCellSize;
//   int iCell = cells[qid * maxNProbe];
//   bool nCellRepeated = false;
//   bool cCellRepeated = false;
//   load_part2_to_cache(part2, part2Cache, iCell);

//   for (int cCell = 0; cCell < nProbe; cCell++){
//     int cCellStart = nCellStart;
//     int cCellSize = nCellSize;
//     int cCellEnd = nCellEnd;
//     if (!cCellRepeated){
//       store_precomputed_to_smem(part1Cache, part2Cache, sMem);
//     }

//     if (cCell < nProbe - 1){
//       int tCellStart = cellStart[qid * maxNProbe + cCell + 1];
//       if (tCellStart != cCellStart){
//         nCellStart = tCellStart;
//         nCellSize = cellSize[qid * maxNProbe + cCell + 1];
//         nCellEnd = nCellStart + nCellSize;
//         iCell = cells[qid * maxNProbe + cCell + 1];
//         load_part2_to_cache(part2, part2Cache, iCell);
//         nCellRepeated = false;
//       } else {
//         nCellRepeated = true;
//       }
//     }
//     if (cCellRepeated){
//       cCellRepeated = nCellRepeated;
//       continue;
//     }
//     cCellRepeated = nCellRepeated;

//     float cBaseSim = baseSims[qid * maxNProbe + cCell];
//     int nIter = (cCellSize + _TPB_ - 1) / _TPB_;
//     for (int iter = 0; iter < nIter; iter++ ){
//       int iN = cCellStart + iter * _TPB_ + tid;
//       int cIsEmpty = 0;
//       if (iter + 1 < nIter){
//         int nIN = cCellStart + (iter + 1) * _TPB_ + tid;
//         if (nIN < cCellEnd){
//           prefetch_data(data, nIN, nData);
//           __prefetch_l2(isEmpty + nIN);
//         }
//       }
//       float value;
//       float index = iN;
//       if (iN < cCellEnd){
//         value = cBaseSim;
//         cIsEmpty = isEmpty[iN];
//         uint8n_t dataCache[_M_ / _NCS_];
//         load_data(data, dataCache, iN, nData);
//         consume_data(sMem, dataCache, value);
//       } else {
//         value = -INFINITY;
//       }
//       value = cIsEmpty == 0 ? value : -INFINITY;
//       index = cIsEmpty == 0 ? index : -1;
//       thread_comparator(finalValue, finalIndex, value, index);
//     }
//   }
//   block_max(finalValue, finalIndex, sMem);

//   if (tid == 0){
//     const int writeAddress = qid;
//     gValue[writeAddress] = finalValue;
//     gIndex[writeAddress] = finalIndex;
//   }
// }