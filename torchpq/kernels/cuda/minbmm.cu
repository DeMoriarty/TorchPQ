__device__ __forceinline__ float atomicMin(float *address, float val)
{
  int ret = __float_as_int(*address);
  while(val < __int_as_float(ret))
  {
    int old = ret;
    if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}

__device__ void min_dim_1(
  float8 cCache[8],
  _VOLATILE_ float valSmem[16][128+4],
  _VOLATILE_ float idxSmem[16][128+4],
  float* values,
  ll_t* indices,
  int gStartx, int gStarty, int tid, int bid,
  int M, int N
){
  int vx = tid % 16;
  int vy = tid / 16;

  #pragma unroll
  for (int ni = 0; ni < 8; ni++){
    // initialize with first value

    float value;
    if (likely(gStarty + vy*8 < M)){
      value = cCache[0].val[ni];
    } else {
      value = INFINITY;
    }
    float index = vy*8;

    // Reduce within thread
    #pragma unroll
    for (int mi = 1; mi < 8; mi++){
      int iM = gStarty + vy*8 + mi;
      float temp;
      if (likely(iM < M)){
        temp = cCache[mi].val[ni];
      } else {
        temp = INFINITY;
      }
      if (temp < value){
        value = temp;
        index = vy*8 + mi;
      }
    }

    // Store reduced values and indices in shared memory
    valSmem[vy][vx * 8 + ni] = value;
    idxSmem[vy][vx * 8 + ni] = index;
  }
  __syncthreads();

  // first 128 threads do block wise reduction
  if (tid < 128){
    float value = valSmem[0][tid];
    float index = idxSmem[0][tid];
    
    #pragma unroll
    for (int i=1; i<16; i++){
      float temp = valSmem[i][tid];
      if (temp < value){
        value = temp;
        index = idxSmem[i][tid];
      }
    }
    
    // global reduction
    int iN = gStartx + tid;
    if (iN < N){
      atomicMin(values + (bid) * N + iN, value);
      if (value <= values[(bid) * N + iN]){
        indices[(bid) * N + iN] = ll_t(index) + gStarty;
      }
    }
    /*
    */
  }
}

__device__ void min_dim_2(
  float8 cCache[8],
  _VOLATILE_ float valSmem[16][128+4],
  _VOLATILE_ float idxSmem[16][128+4],
  float* values,
  ll_t* indices,
  int gStartx, int gStarty, int tid, int bid,
  int M, int N
){
  int vx = tid % 16;
  int vy = tid / 16;

  #pragma unroll
  for (int mi = 0; mi < 8; mi++){
    // initialize with first value
    float value;
    if (likely(gStartx + vx*8 < N)){
      value = cCache[mi].val[0];
    } else {
      value = INFINITY;
    }
    float index = vx*8;

    // Reduce within thread
    #pragma unroll
    for (int ni = 1; ni < 8; ni++){
      int iN = gStartx + vx*8 + ni;
      float temp;
      if (likely(iN < N)){
        temp = cCache[mi].val[ni];
      } else {
        temp = INFINITY;
      }
      if (temp < value){
        value = temp;
        index = vx*8 + ni;
      }
    }

    // Store reduced values and indices in shared memory
    valSmem[vx][vy * 8 + mi] = value;
    idxSmem[vx][vy * 8 + mi] = index;
  }
  __syncthreads();

  // first 128 threads do block-wise reduction
  if (tid < 128){
    float value = valSmem[0][tid];
    float index = idxSmem[0][tid];
    #pragma unroll
    for (int i = 1; i < 16; i++){
      float temp = valSmem[i][tid];
      if (temp < value){
        value = temp;
        index = idxSmem[i][tid];
      }
    }

    // global reduction
    int iM = gStarty + tid;
    if (iM < M){
      atomicMin(values + (bid) * M + iM, value);
      if (value <= values[(bid) * M + iM]){
        indices[(bid) * M + iM] = ll_t(index) + gStartx;
      }
    }
  }
}

extern "C"
__global__ void min_bmm_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
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
  #pragma unroll
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

  // Reduce along DIM
  if (DIM == 1){
    min_dim_1(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  } else if (DIM == 2){
    min_dim_2(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  }
}

extern "C"
__global__ void min_bmm_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
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
  #pragma unroll
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

  // Reduce along DIM
  if (DIM == 1){
    min_dim_1(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  } else if (DIM == 2){
    min_dim_2(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  }
}

extern "C"
__global__ void min_bmm_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
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

  // Reduce along DIM
  if (DIM == 1){
    min_dim_1(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  } else if (DIM == 2){
    min_dim_2(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  }
}

extern "C"
__global__ void min_bmm_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ values,
  ll_t* __restrict__ indices,
  int M, int N, int K, int DIM
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
  #pragma unroll
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

  // Reduce along DIM
  if (DIM == 1){
    min_dim_1(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  } else if (DIM == 2){
    min_dim_2(
      cCache, aSmem, bSmem, values, indices,
      gStartx, gStarty, tid, bid, M, N);
  }
}