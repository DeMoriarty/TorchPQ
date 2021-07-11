#define _VOLATILE_  

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)

typedef long long ll_t;
typedef unsigned long long ull_t;
typedef unsigned char uint8_t;

typedef struct __builtin_align__(32) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
} _float8;

typedef union {
  _float8 f8;
  float val[8];
} float8;

__device__ void init_cCache(
  float8 cCache[8]
) {
  #pragma unroll
  for (int i=0; i<8; i++){
    #pragma unroll
    for (int j=0; j<8; j++){
      cCache[i].val[j] = 0.f;
    }
  }
}

__device__ void thread_matmul_v4(
  _VOLATILE_ float aSM[8][128+4],
  _VOLATILE_ float bSM[8][128+4],
  float8 cCache[8],
  int vx, int vy
) {
  float aCache1[8];
  float aCache2[8];
  #pragma unroll
  for (int mi=0; mi<8; mi++){
    aCache1[mi] = aSM[0][8*vy + mi];
  }

  #pragma unroll
  for (int ki=0; ki<8; ki++){
    int is_odd = ki & 1;
    if (is_odd == 0){
      if (likely(ki < 7)){
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          aCache2[mi] = aSM[ki+1][8*vy + mi];
        }
      }
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        float b = bSM[ki][vx/4 + 8*vx + ni];
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          float a = aCache1[mi];
          cCache[mi].val[ni] = fmaf(a, b, cCache[mi].val[ni]);
        }
      }
    } else {
      if (likely(ki < 7)){
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          aCache1[mi] = aSM[ki+1][8*vy + mi];
        }
      }
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        float b = bSM[ki][vx/4 + 8*vx + ni];
        #pragma unroll
        for (int mi=0; mi<8; mi++){
          float a = aCache2[mi];
          cCache[mi].val[ni] = fmaf(a, b, cCache[mi].val[ni]);
        }
      }
    }
  }
}

__device__ void thread_matmul_v3(
  _VOLATILE_ float aSM[8][128+4],
  _VOLATILE_ float bSM[8][128+4],
  float8 cCache[8],
  int vx, int vy
) {
  float aCache[8];

  #pragma unroll
  for (int ki=0; ki<8; ki++){
    #pragma unroll
    for (int mi=0; mi<8; mi++){
      aCache[mi] = aSM[ki][8*vy + mi];
    }
    #pragma unroll
    for (int ni=0; ni<8; ni++){
      float b = bSM[ki][vx/4 + 8*vx + ni];
      #pragma unroll
      for (int mi=0; mi<8; mi++){
        float a = aCache[mi];
        cCache[mi].val[ni] = fmaf(a, b, cCache[mi].val[ni]);
      }
    }
  }
}

__device__ void mask_cCache(
  float8 cCache[8],
  const uint8_t* ElementMask,
  int gStartx,
  int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
  #pragma unroll
  for (int i=0; i<8; i++){
    int iM = gStarty + vy*8 + i;
    if (likely(iM < M)){
      #pragma unroll
      for (int j=0; j<8; j++){
        int iN = gStartx + vx*8 + j;
        if (likely(iN < N)){
          uint8_t element_mask = ElementMask[(__MASK_BID__)*M*N + (iM)*N + (iN)];
          cCache[i].val[j] *= element_mask;
        }
      }
    }
  }
}

// Unsafe
__device__ void write_c(
  float8 cCache[8],
  float* C,
  int gStartx, int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
  #pragma unroll
  for (int i=0; i<8; i++){
    int iM = gStarty + vy*8 + i;
    if (likely(iM < M)){
      int iN_start = gStartx + vx*8;
      reinterpret_cast<float8*>(C + (bid)*M*N + (iM)*N + (iN_start))[0] = cCache[i];
    }
  }
}

__device__ void write_c_v3(
  float8 cCache[8],
  float* C,
  int gStartx, int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
  __shared__ volatile float cSM[16][128];
  #pragma unroll
  for (int mi=0; mi<8; mi++){
    int iM = gStarty + vy*8 + mi;
    // Store 1 row from cCache to cSM
    if (iM < M){
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        cSM[vy][vx*8 + ni] = cCache[mi].val[ni];
      }
      // Store to C
      #pragma unroll
      for (int ni=0; ni<8; ni++){
        int iN = gStartx + 16*ni + vx;
        if (iN < N){
          float cVal = cSM[vy][16*ni + vx];
          //store(C+(bid)*M*N + (iM)*N + (iN), cVal);
          C[(bid)*M*N + (iM)*N + (iN)] = cVal;
        }
      }
    }
  } 
}

extern "C"
__global__ void mbmm_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const uint8_t* __restrict__ BlockMask,
  const uint8_t* __restrict__ ThreadMask,
  const uint8_t* __restrict__ ElementMask,
  int M, int N, int K
){
}

extern "C"
__global__ void mbmm_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const uint8_t* __restrict__ BlockMask,
  const uint8_t* __restrict__ ThreadMask,
  const uint8_t* __restrict__ ElementMask,
  int M, int N, int K
){
}

extern "C"
__global__ void mbmm_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const uint8_t* __restrict__ BlockMask,
  const uint8_t* __restrict__ ThreadMask,
  const uint8_t* __restrict__ ElementMask,
  int M, int N, int K
){
  int tid = threadIdx.x;     // thread idx
  int bid = blockIdx.z;      // batch idx

  // Neighboring blocks are grouped into PN x PM block groups in order to increase
  // L1 cache hit rate.
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

  int bM = (M + 128 - 1) / 128;
  int bN = (N + 128 - 1) / 128;
  int tM = (M + 8 - 1) / 8;
  int tN = (N + 8 - 1) / 8;
  uint8_t block_mask = BlockMask[__MASK_BID__*bM*bN + (bIdxY)*bN + (bIdxX)];
  uint8_t thread_mask = ThreadMask[__MASK_BID__*tM*tN + (bIdxY*16 + vy)*tN + (bIdxX*16 + vx) ];
  if (block_mask == 0){
    return;
  }

  __shared__ _VOLATILE_ float aSM1[8][128+4];
  __shared__ _VOLATILE_ float bSM1[8][128+4];
  __shared__ _VOLATILE_ float aSM2[8][128+4];
  __shared__ _VOLATILE_ float bSM2[8][128+4];
  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  // Load initial 16 x 128 tile of A and B to buffer1 and buffer2
  #pragma unroll
  for (int i=0; i<4; i++){
    int iM = gStarty + dy + i*32;
    int iN = gStartx + wx + i*32;
    if (likely(iM < M)){
      if (likely(dx < K)){
        aBuffer1[i] = load(A + (bid)*M*K + (iM)*K + (dx));
      } else {
        aBuffer1[i] = 0.f;
      }
      if (likely(dx+8 < K)){
        aBuffer2[i] = load(A + (bid)*M*K + (iM)*K + (dx+8));
      } else {
        aBuffer2[i] = 0.f;
      }
    }
    if (likely(iN < N)){
      if (likely(wy < K)){
        bBuffer1[i] = load(B + (bid)*N*K + (wy)*N + (iN));
      } else {
        bBuffer1[i] = 0.f;
      }
      if (likely(wy+8 < K)){
        bBuffer2[i] = load(B + (bid)*N*K + (wy+8)*N + (iN));
      } else {
        bBuffer2[i] = 0.f;
      }
    }
  }

  // Number of main loop iterations is ceil(k/16)
  int nIt = (K + 16 - 1) / 16;
  #pragma unroll
  for (int itr=0; itr<nIt; itr++){
    int gStartk = itr * 16;

    // Index on K axis of A and B
    int iKA = gStartk + 16 + dx;
    int iKB = gStartk + 16 + wy;

    #pragma unroll
    for (int i=0; i<4; i++){
      // Store buffered tiles into shared memory
      aSM1[dx][dy+i*32] = aBuffer1[i];
      bSM1[wy][wx+i*32+i] = bBuffer1[i];
      aSM2[dx][dy+i*32] = aBuffer2[i];
      bSM2[wy][wx+i*32+i] = bBuffer2[i];

      // Start loading next 16*128 tile of A and B to buffer1 and buffer2.
      // Don't load anything on the last iteration.
      // Loading from global memory will not block thread_matmul
      if (likely(itr < nIt - 1)){
        int iM = gStarty + i*32 + dy;
        int iN = gStartx + i*32 + wx;
        
        if (likely(iM < M)){
          if (likely(iKA < K)){
            aBuffer1[i] = load(A + (bid)*M*K + (iM)*K + (iKA));
          } else {
            aBuffer1[i] = 0.f;
          }
          if (likely(iKA+8 < K)){
            aBuffer2[i] = load(A + (bid)*M*K + (iM)*K + (iKA+8));
          } else {
            aBuffer2[i] = 0.f;
          }
        }

        if (likely(iN < N)){
          if (likely(iKB < K)){
            bBuffer1[i] = load(B + (bid)*N*K + (iKB)*N + (iN));
          } else {
            bBuffer1[i] = 0.f;
          }
          if (likely(iKB+8 < K)){
            bBuffer2[i] = load(B + (bid)*N*K + (iKB+8)*N + (iN));
          } else {
            bBuffer2[i] = 0.f;
          }
        }
      }
    }
    // synchroznie threads in order to make sure tiles of A and B are fully
    // loaded to shared memory.
    __syncthreads();

    // Each thread computes 8 x 8 matrix multiplication
    // Accumulating intermediate results in cCache
    // aSM1, bSM1, aSM2, bSM2 are consumed
    
    if (thread_mask != 0){
      thread_matmul_v3(aSM1, bSM1, cCache, vx, vy);
      thread_matmul_v3(aSM2, bSM2, cCache, vx, vy);
    }

    // synchronize threads to signal that shared memory is consumed.
    __syncthreads();
  }
  
  // At the end of main loop, store cCache to C
  if (0 < thread_mask < 64){
    mask_cCache(cCache, ElementMask, gStartx, gStarty, vx, vy, bid, M, N);
  }
  write_c_v3(cCache, C, gStartx, gStarty, vx, vy, bid, M, N);
}

extern "C"
__global__ void mbmm_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const uint8_t* __restrict__ BlockMask,
  const uint8_t* __restrict__ ThreadMask,
  const uint8_t* __restrict__ ElementMask,
  int M, int N, int K
){
}