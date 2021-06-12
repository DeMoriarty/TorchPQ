extern "C"
__global__ void bmm_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int M, int N, int K
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

  __shared__ _VOLATILE_ float aSmem1[8][128+4];
  __shared__ _VOLATILE_ float bSmem1[8][128+4];
  __shared__ _VOLATILE_ float aSmem2[8][128+4];
  __shared__ _VOLATILE_ float bSmem2[8][128+4];
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

    #pragma unroll
    buffer2smem_tn(
      aSmem1, aSmem2, bSmem1, bSmem2,
      aBuffer1, aBuffer2, bBuffer1, bBuffer2
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

    // Each thread computes 8 x 8 matrix multiplication
    // Accumulate intermediate results in cCache
    // aSmem1, bSmem1, aSmem2, bSmem2 are consumed
    thread_matmul_v3(aSmem1, bSmem1, cCache, vx, vy);
    thread_matmul_v3(aSmem2, bSmem2, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }
  
  // At the end of main loop, store cCache to C
  //write_c(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
  write_c_v3(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
}

extern "C"
__global__ void bmm_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int M, int N, int K
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
  } K
  // These are used to re-arrange threads into different shapes
  // for example: (256) -> (16, 16) -> (8, 32) -> (32, 8)
  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ _VOLATILE_ float aSmem1[8][128+4];
  __shared__ _VOLATILE_ float bSmem1[8][128+4];
  __shared__ _VOLATILE_ float aSmem2[8][128+4];
  __shared__ _VOLATILE_ float bSmem2[8][128+4];
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

    buffer2smem_nt(
      aSmem1, aSmem2, bSmem1, bSmem2,
      aBuffer1, aBuffer2, bBuffer1, bBuffer2
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

    // Each thread computes 8 x 8 matrix multiplication
    // Accumulate intermediate results in cCache
    // aSmem1, bSmem1, aSmem2, bSmem2 are consumed
    thread_matmul_v3(aSmem1, bSmem1, cCache, vx, vy);
    thread_matmul_v3(aSmem2, bSmem2, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }
  
  // At the end of main loop, store cCache to C
  //write_c(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
  write_c_v3(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
}

extern "C"
__global__ void bmm_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int M, int N, int K
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

  __shared__ _VOLATILE_ float aSmem1[8][128+4];
  __shared__ _VOLATILE_ float bSmem1[8][128+4];
  __shared__ _VOLATILE_ float aSmem2[8][128+4];
  __shared__ _VOLATILE_ float bSmem2[8][128+4];
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

    #pragma unroll
    buffer2smem_nn(
      aSmem1, aSmem2, bSmem1, bSmem2,
      aBuffer1, aBuffer2, bBuffer1, bBuffer2
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

    // Each thread computes 8 x 8 matrix multiplication
    // Accumulate intermediate results in cCache
    // aSmem1, bSmem1, aSmem2, bSmem2 are consumed
    thread_matmul_v3(aSmem1, bSmem1, cCache, vx, vy);
    thread_matmul_v3(aSmem2, bSmem2, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }
  
  // At the end of main loop, store cCache to C
  //write_c(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
  write_c_v3(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
}

extern "C"
__global__ void bmm_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int M, int N, int K
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

  __shared__ _VOLATILE_ float aSmem1[8][128+4];
  __shared__ _VOLATILE_ float bSmem1[8][128+4];
  __shared__ _VOLATILE_ float aSmem2[8][128+4];
  __shared__ _VOLATILE_ float bSmem2[8][128+4];
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

    #pragma unroll
    buffer2smem_tt(
      aSmem1, aSmem2, bSmem1, bSmem2,
      aBuffer1, aBuffer2, bBuffer1, bBuffer2
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

    // Each thread computes 8 x 8 matrix multiplication
    // Accumulate intermediate results in cCache
    // aSmem1, bSmem1, aSmem2, bSmem2 are consumed
    thread_matmul_v3(aSmem1, bSmem1, cCache, vx, vy);
    thread_matmul_v3(aSmem2, bSmem2, cCache, vx, vy);

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }
  
  // At the end of main loop, store cCache to C
  //write_c(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
  write_c_v3(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
}