#define isnan(x) ( x != x )
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

__device__ void mutex_lock(
  unsigned int *mutex
) {
  unsigned int ns = 8;
  unsigned int counter = 0;
  __syncthreads();
  if (threadIdx.x == 0 ){
    while (atomicCAS(mutex, 0, 1) == 1) {
      __nanosleep(ns);
      counter ++;
      if (counter > 100000) break;
      if (ns < 256) {
        ns *= 2;
      }
    }
  }
  __syncthreads();
}

__device__ void mutex_lock_noop(
) {
  __syncthreads();
}

__device__ void mutex_unlock(
  unsigned int *mutex
) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0){
    atomicExch(mutex, 0);
    __threadfence();
  }
  __syncthreads();
}

__device__ void mutex_unlock_noop(){
  __syncthreads();
  __syncthreads();
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
  const float other_value = __shfl_xor_sync(0xFFFFFFFF, value, stride);
  const float other_index = __shfl_xor_sync(0xFFFFFFFF, index, stride);
  bool condition = value < other_value == direction;
  index = condition ? other_index : index;
  value = condition ? other_value : value;
}

__device__ __forceinline__ void block_comparator(
  float &value,
  float &index,
  const int stride,
  const int direction,
  const int laneID,
  _VOLATILE_ float valSmem[128+4],
  _VOLATILE_ float idxSmem[128+4]
){
  valSmem[laneID] = value;
  idxSmem[laneID] = index;
  __syncthreads();

  float other_value = valSmem[laneID ^ stride];
  float other_index = idxSmem[laneID ^ stride];
  __syncthreads();

  bool condition = value < other_value == direction;
  index = condition ? other_index : index;
  value = condition ? other_value : value;
}

__device__ void bitonic_sort_128(
  float &value,
  float &index,
  _VOLATILE_ float valSmem[128+4],
  _VOLATILE_ float idxSmem[128+4]
) {
  unsigned int laneID = threadIdx.x % 128;
  warp_comparator(value, index, 1, bfe(laneID, 1) ^ bfe(laneID, 0));

  warp_comparator(value, index, 2, bfe(laneID, 2) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 2) ^ bfe(laneID, 0));

  warp_comparator(value, index, 4, bfe(laneID, 3) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 3) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 3) ^ bfe(laneID, 0));

  warp_comparator(value, index, 8, bfe(laneID, 4) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 4) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 4) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 4) ^ bfe(laneID, 0));

  warp_comparator(value, index, 16, bfe(laneID, 5) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 5) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 5) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 5) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 5) ^ bfe(laneID, 0));

  block_comparator(value, index, 32, bfe(laneID, 6) ^ bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 6) ^ bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 6) ^ bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 6) ^ bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 6) ^ bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 6) ^ bfe(laneID, 0));

  block_comparator(value, index, 64, bfe(laneID, 6), laneID, valSmem, idxSmem);
  block_comparator(value, index, 32, bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, bfe(laneID, 4));
  warp_comparator(value, index, 8, bfe(laneID, 3));
  warp_comparator(value, index, 4, bfe(laneID, 2));
  warp_comparator(value, index, 2, bfe(laneID, 1));
  warp_comparator(value, index, 1, bfe(laneID, 0));
}

__device__ void bitonic_sort_256(
  float &value,
  float &index,
  float* g_values,
  ll_t* g_indices,
  float valSmem[128+4],
  float idxSmem[128+4],
  int Q, int adr, bool ok
){
  int laneID = threadIdx.x % 128;
  float other_index;
  float other_value; 
  if (ok){
    other_value = g_values[adr];
    other_index = g_indices[adr];
  } else {
    other_value = -INFINITY;
    other_index = 0;
  }
  bool condition = value > other_value == 0;
  if (condition){
    value = value + other_value;
    index = index + other_index;
    other_value = value - other_value;
    other_index = index - other_index;
    value = value - other_value;
    index = index - other_index;
  }

  block_comparator(value, index, 64, !bfe(laneID, 6), laneID, valSmem, idxSmem);
  block_comparator(value, index, 32, !bfe(laneID, 5), laneID, valSmem, idxSmem);
  warp_comparator(value, index, 16, !bfe(laneID, 4));
  warp_comparator(value, index, 8, !bfe(laneID, 3));
  warp_comparator(value, index, 4, !bfe(laneID, 2));
  warp_comparator(value, index, 2, !bfe(laneID, 1));
  warp_comparator(value, index, 1, !bfe(laneID, 0));
  /*
  */
  if (ok){
    g_values[adr] = value;
    g_indices[adr] = index;
  }
}

__device__ void bitonicSort256_noop()
{
  __syncthreads();
  __syncthreads();
  __syncthreads();
  __syncthreads();
}

__device__ void topk_dim_1(
  float8 cCache[8],
  _VOLATILE_ float valSmem[16][128+4],
  _VOLATILE_ float idxSmem[16][128+4],
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int gStartx, int gStarty, int bid,
  int M, int N, int Q
){
  int tid = threadIdx.x;
  int vx = tid % 16;
  int vy = tid / 16;
  int hx = tid % 128;
  int hy = tid / 128;
  #pragma unroll
  for (int ni=0; ni<8; ni++){
    int iN = gStartx + vx*8 + ni;
    //if (iN < N) break;

    // Store cCache to cSM
    #pragma unroll
    for (int mi=0; mi<8; mi++){
      int iM = gStarty + vy*8 + mi;
      if (likely(iM < M && iN < N)){
        valSmem[vx][vy*8 + mi] = cCache[mi].val[ni];
        idxSmem[vx][vy*8 + mi] = iM;
      } else {
        valSmem[vx][vy*8 + mi] = -INFINITY;
        idxSmem[vx][vy*8 + mi] = -1;
      }
    }
    __syncthreads();
    // Load from cSM to cCache
    #pragma unroll
    for (int i=0; i<8; i++){
      float value = valSmem[hy*8 + i][hx];
      float index = idxSmem[hy*8 + i][hx];
      bitonic_sort_128(
        value, index,
        valSmem[hy*8 + i], idxSmem[hy*8 + i]
      );
      int iN = gStartx + (hy*8 + i)*8 + ni;
      int adr = (bid)*N*Q + iN*Q + hx;
      mutex_lock( &mutex[(bid)*N + iN] );
      bitonic_sort_256(
        value, index, 
        values, indices, 
        valSmem[hy*8+i], idxSmem[hy*8+i],
        Q, adr, iN < N
      );
      mutex_unlock( &mutex[(bid)*N + iN] );
    }
  }
}

__device__ void topk_dim_2(
  float8 cCache[8],
  _VOLATILE_ float valSmem[16][128+4],
  _VOLATILE_ float idxSmem[16][128+4],
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int gStartx, int gStarty, int bid,
  int M, int N, int Q
){
  int tid = threadIdx.x;
  int vx = tid % 16;
  int vy = tid / 16;
  int hx = tid % 128;
  int hy = tid / 128;
  #pragma unroll
  for (int mi=0; mi<8; mi++){
    int iM = gStarty + vy*8 + mi;
    //if (iM >= M) break;

    // Store cCache to cSM
    #pragma unroll
    for (int ni=0; ni<8; ni++){
      int iN = gStartx + vx*8 + ni;
      if (likely(iN < N && iM < M)){
        valSmem[vy][vx*8 + ni] = cCache[mi].val[ni];
        idxSmem[vy][vx*8 + ni] = iN;
      } else {
        valSmem[vy][vx*8 + ni] = -INFINITY;
        idxSmem[vy][vx*8 + ni] = -1;
      }
    }
    __syncthreads();
    // Load from cSM to cCache
    #pragma unroll
    for (int i=0; i<8; i++){
      float value = valSmem[hy*8 + i][hx];
      float index = idxSmem[hy*8 + i][hx];
      bitonic_sort_128(
        value, index,
        valSmem[hy*8 + i], idxSmem[hy*8 + i]
      );
      int iM = gStarty + (hy*8 + i)*8 + mi;
      int adr = (bid)*M*Q + iM*Q + hx;
      mutex_lock( &mutex[(bid)*M + iM] );
      bitonic_sort_256(
        value, index, 
        values, indices, 
        valSmem[hy*8+i], idxSmem[hy*8+i],
        Q, adr, iM < M
      );
      mutex_unlock( &mutex[(bid)*M + iM] );
    }
  }
}

extern "C"
__global__ void topk_bmm_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int M, int N, int K, int DIM, int Q
){
  int tid = threadIdx.x;     // thread idx
  int bid = blockIdx.z;      // batch idx

  // Neighboring blocks are grouped into PN x PM block groups in order to increase
  // L1 cache hit rate
  // There are ceil(M/PM) x ceil(N/PN) block groups in total.
  // Blocks within block groups are indexed with blockIdx.x % PN and blockIdx.x / PN
  int px = blockIdx.x % _PN_;
  int py = blockIdx.x / _PN_;
  int bDimX = (N + (128*_PN_) - 1) / (128*_PN_); 
  int bDimY = (M + (128*_PM_) - 1) / (128*_PM_); 
  int bIdxX = (blockIdx.y % bDimX) * _PN_ + px;
  int bIdxY = (blockIdx.y / bDimX) * _PM_ + py;
  int gStartx = bIdxX * 128;   // starting index of block on N axis
  int gStarty = bIdxY * 128;   // starting index of block on M axis
  if (gStartx > N || gStarty > M){
    return;
  }
  // These are used to re-arrange threads into different shapes
  // for example: (256) -> (16, 16) -> (8, 32) -> (32, 8)
  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ _VOLATILE_ float aSmem[16][128+4];
  __shared__ _VOLATILE_ float bSmem[16][128+4];

  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  // Load initial 16 x 128 tile of A and B to buffer1 and buffer2
  load_ab_tn(
    A, B, 
    aBuffer1, aBuffer2, bBuffer1, bBuffer2,
    bid, gStartx, gStarty, 0,
    M, N, K
  );

  // Number of main loop iterations is ceil(k/16)
  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for (int itr=0; itr<nIt; itr++){
    int gStartk = itr * 16;
    buffer2smem_16_tn(
      aSmem, bSmem,
      aBuffer1, aBuffer2,
      bBuffer1, bBuffer2
    );
    if (likely(itr < nIt - 1)){
      load_ab_tn(
        A, B, 
        aBuffer1, aBuffer2, bBuffer1, bBuffer2,
        bid, gStartx, gStarty, gStartk + 16,
        M, N, K
      );
    }
    // synchroznie threads in order make sure tiles of A and B are fully
    // loaded to shared memory.
    __syncthreads();

    thread_matmul_16_v3(aSmem, bSmem, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }

  // TopK sort along DIM
  if (DIM == 1){
    topk_dim_1(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  } else if (DIM == 2){
    topk_dim_2(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  }
}

extern "C"
__global__ void topk_bmm_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int M, int N, int K, int DIM, int Q
){
  int tid = threadIdx.x;     // thread idx
  int bid = blockIdx.z;      // batch idx

  // Neighboring blocks are grouped into PN x PM block groups in order to increase
  // L1 cache hit rate
  // There are ceil(M/PM) x ceil(N/PN) block groups in total.
  // Blocks within block groups are indexed with blockIdx.x % PN and blockIdx.x / PN
  int px = blockIdx.x % _PN_;
  int py = blockIdx.x / _PN_;
  int bDimX = (N + (128*_PN_) - 1) / (128*_PN_); 
  int bDimY = (M + (128*_PM_) - 1) / (128*_PM_); 
  int bIdxX = (blockIdx.y % bDimX) * _PN_ + px;
  int bIdxY = (blockIdx.y / bDimX) * _PM_ + py;
  int gStartx = bIdxX * 128;   // starting index of block on N axis
  int gStarty = bIdxY * 128;   // starting index of block on M axis
  if (gStartx > N || gStarty > M){
    return;
  }
  // These are used to re-arrange threads into different shapes
  // for example: (256) -> (16, 16) -> (8, 32) -> (32, 8)
  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ _VOLATILE_ float aSmem[16][128+4];
  __shared__ _VOLATILE_ float bSmem[16][128+4];

  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  // Load initial 16 x 128 tile of A and B to buffer1 and buffer2
  load_ab_nt(
    A, B, 
    aBuffer1, aBuffer2, bBuffer1, bBuffer2,
    bid, gStartx, gStarty, 0,
    M, N, K
  );

  // Number of main loop iterations is ceil(k/16)
  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for (int itr=0; itr<nIt; itr++){
    int gStartk = itr * 16;
    buffer2smem_16_nt(
      aSmem, bSmem,
      aBuffer1, aBuffer2,
      bBuffer1, bBuffer2
    );
    if (likely(itr < nIt - 1)){
      load_ab_nt(
        A, B, 
        aBuffer1, aBuffer2, bBuffer1, bBuffer2,
        bid, gStartx, gStarty, gStartk + 16,
        M, N, K
      );
    }
    // synchroznie threads in order make sure tiles of A and B are fully
    // loaded to shared memory.
    __syncthreads();

    thread_matmul_16_v3(aSmem, bSmem, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }

  // TopK sort along DIM
  if (DIM == 1){
    topk_dim_1(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  } else if (DIM == 2){
    topk_dim_2(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  }
}

extern "C"
__global__ void topk_bmm_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int M, int N, int K, int DIM, int Q
){
  int tid = threadIdx.x;     // thread idx
  int bid = blockIdx.z;      // batch idx

  // Neighboring blocks are grouped into PN x PM block groups in order to increase
  // L1 cache hit rate
  // There are ceil(M/PM) x ceil(N/PN) block groups in total.
  // Blocks within block groups are indexed with blockIdx.x % PN and blockIdx.x / PN
  int px = blockIdx.x % _PN_;
  int py = blockIdx.x / _PN_;
  int bDimX = (N + (128*_PN_) - 1) / (128*_PN_); 
  int bDimY = (M + (128*_PM_) - 1) / (128*_PM_); 
  int bIdxX = (blockIdx.y % bDimX) * _PN_ + px;
  int bIdxY = (blockIdx.y / bDimX) * _PM_ + py;
  int gStartx = bIdxX * 128;   // starting index of block on N axis
  int gStarty = bIdxY * 128;   // starting index of block on M axis
  if (gStartx > N || gStarty > M){
    return;
  }
  // These are used to re-arrange threads into different shapes
  // for example: (256) -> (16, 16) -> (8, 32) -> (32, 8)
  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ _VOLATILE_ float aSmem[16][128+4];
  __shared__ _VOLATILE_ float bSmem[16][128+4];

  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  // Load initial 16 x 128 tile of A and B to buffer1 and buffer2
  load_ab_nn(
    A, B, 
    aBuffer1, aBuffer2, bBuffer1, bBuffer2,
    bid, gStartx, gStarty, 0,
    M, N, K
  );

  // Number of main loop iterations is ceil(k/16)
  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for (int itr=0; itr<nIt; itr++){
    int gStartk = itr * 16;
    buffer2smem_16_nn(
      aSmem, bSmem,
      aBuffer1, aBuffer2,
      bBuffer1, bBuffer2
    );
    if (likely(itr < nIt - 1)){
      load_ab_nn(
        A, B, 
        aBuffer1, aBuffer2, bBuffer1, bBuffer2,
        bid, gStartx, gStarty, gStartk + 16,
        M, N, K
      );
    }
    // synchroznie threads in order make sure tiles of A and B are fully
    // loaded to shared memory.
    __syncthreads();

    thread_matmul_16_v3(aSmem, bSmem, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }

  // TopK sort along DIM
  if (DIM == 1){
    topk_dim_1(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  } else if (DIM == 2){
    topk_dim_2(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  }
}

extern "C"
__global__ void topk_bmm_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* values,
  ll_t* indices,
  unsigned int* mutex,
  int M, int N, int K, int DIM, int Q
){
  int tid = threadIdx.x;     // thread idx
  int bid = blockIdx.z;      // batch idx

  // Neighboring blocks are grouped into PN x PM block groups in order to increase
  // L1 cache hit rate
  // There are ceil(M/PM) x ceil(N/PN) block groups in total.
  // Blocks within block groups are indexed with blockIdx.x % PN and blockIdx.x / PN
  int px = blockIdx.x % _PN_;
  int py = blockIdx.x / _PN_;
  int bDimX = (N + (128*_PN_) - 1) / (128*_PN_); 
  int bDimY = (M + (128*_PM_) - 1) / (128*_PM_); 
  int bIdxX = (blockIdx.y % bDimX) * _PN_ + px;
  int bIdxY = (blockIdx.y / bDimX) * _PM_ + py;
  int gStartx = bIdxX * 128;   // starting index of block on N axis
  int gStarty = bIdxY * 128;   // starting index of block on M axis
  if (gStartx > N || gStarty > M){
    return;
  }
  // These are used to re-arrange threads into different shapes
  // for example: (256) -> (16, 16) -> (8, 32) -> (32, 8)
  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ _VOLATILE_ float aSmem[16][128+4];
  __shared__ _VOLATILE_ float bSmem[16][128+4];

  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  // Load initial 16 x 128 tile of A and B to buffer1 and buffer2
  load_ab_tt(
    A, B, 
    aBuffer1, aBuffer2, bBuffer1, bBuffer2,
    bid, gStartx, gStarty, 0,
    M, N, K
  );

  // Number of main loop iterations is ceil(k/16)
  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for (int itr=0; itr<nIt; itr++){
    int gStartk = itr * 16;
    buffer2smem_16_tt(
      aSmem, bSmem,
      aBuffer1, aBuffer2,
      bBuffer1, bBuffer2
    );
    if (likely(itr < nIt - 1)){
      load_ab_tt(
        A, B, 
        aBuffer1, aBuffer2, bBuffer1, bBuffer2,
        bid, gStartx, gStarty, gStartk + 16,
        M, N, K
      );
    }
    // synchroznie threads in order make sure tiles of A and B are fully
    // loaded to shared memory.
    __syncthreads();

    thread_matmul_16_v3(aSmem, bSmem, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }

  // TopK sort along DIM
  if (DIM == 1){
    topk_dim_1(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  } else if (DIM == 2){
    topk_dim_2(
      cCache, aSmem, bSmem,
      values, indices, mutex,
      gStartx, gStarty, bid, M, N, Q);
  }
}