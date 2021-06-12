#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)

#define _VOLATILE_  

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
#define load(x)        __ldcg(x)
#define store(x, value) __stcs(x, value)

typedef long long ll_t;
typedef unsigned long long ull_t;

typedef struct __builtin_align__(32) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
} _float8;

typedef union {
  _float8 f8;
  float val[8];
} float8;

__device__ void madd(
  float a,
  float b,
  float &c
) {
  c = fmaf(a, b, c);
}

__device__ void squared_l2(
  float a,
  float b,
  float &c
){
  float dif = a - b;
  c = fmaf(dif, dif, c);
}

__device__ void negative_squared_l2(
  float a,
  float b,
  float &c
){
  float dif = a - b;
  c = fmaf(-dif, dif, c);
}

__device__ void l1(
  float a,
  float b,
  float &c
){
  c += fabsf(a - b);
}

__device__ void negative_l1(
  float a,
  float b,
  float &c
){
  c -= fabsf(a - b);
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

__device__ void thread_matmul_16_v3(
  _VOLATILE_ float aSM[16][128+4],
  _VOLATILE_ float bSM[16][128+4],
  float8 cCache[8],
  int vx, int vy
) {
  float aCache[8];

  #pragma unroll
  for (int ki=0; ki<16; ki++){
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
        __DISTANCE_FN__(a, b, cCache[mi].val[ni]);
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
        __DISTANCE_FN__(a, b, cCache[mi].val[ni]);
      }
    }
  }
}

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
      /*
      if (likely(iN_start + 7 < N)){
        reinterpret_cast<float8*>(C + (bid)*M*N + (iM)*N + (iN_start))[0] = cCache[i];
      } else {
        #pragma unroll
        for (int j=0; j<8; j++){
          int iN = iN_start + j;
          if (iN < N){
            C[(bid)*M*N + (iM)*N + (iN)] = cCache[i].val[j];
          }
        }
      }
      */
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
          store(C+(bid)*M*N + (iM)*N + (iN), cVal);
        }
      }
    }
  } 
}

__device__ void load_ab_nn(
  const float* A,
  const float* B,
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4],
  int bid, int gStartx, int gStarty, int gStartk,
  int M, int N, int K
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  int iKA = gStartk + dx;
  int iKB = gStartk + wy;
  #pragma unroll
  for (int i=0; i<4; i++){
    int iM = gStarty + dy + i*32;
    int iN = gStartx + wx + i*32;
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

__device__ void load_ab_tt(
  const float* A,
  const float* B,
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4],
  int bid, int gStartx, int gStarty, int gStartk,
  int M, int N, int K
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  int iKA = gStartk + wy;
  int iKB = gStartk + dx;
  #pragma unroll
  for (int i=0; i<4; i++){
    int iM = gStarty + wx + i*32;
    int iN = gStartx + dy + i*32;
    if (likely(iM < M)){
      if (likely(iKA < K)){
        aBuffer1[i] = load(A + (bid)*M*K + (iKA)*M + (iM));
      } else {
        aBuffer1[i] = 0.f;
      }
      if (likely(iKA+8 < K)){
        aBuffer2[i] = load(A + (bid)*M*K + (iKA+8)*M + (iM));
      } else {
        aBuffer2[i] = 0.f;
      }
    }
    if (likely(iN < N)){
      if (likely(iKB < K)){
        bBuffer1[i] = load(B + (bid)*N*K + (iN)*K + (iKB));
      } else {
        bBuffer1[i] = 0.f;
      }
      if (likely(iKB+8 < K)){
        bBuffer2[i] = load(B + (bid)*N*K + (iN)*K + (iKB+8));
      } else {
        bBuffer2[i] = 0.f;
      }
    }
  }
}

__device__ void load_ab_nt(
  const float* A,
  const float* B,
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4],
  int bid, int gStartx, int gStarty, int gStartk,
  int M, int N, int K
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  int iKA = gStartk + dx;
  int iKB = gStartk + dx;
  #pragma unroll
  for (int i=0; i<4; i++){
    int iM = gStarty + dy + i*32;
    int iN = gStartx + dy + i*32;
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
        bBuffer1[i] = load(B + (bid)*N*K + (iN)*K + (iKB));
      } else {
        bBuffer1[i] = 0.f;
      }
      if (likely(iKB+8 < K)){
        bBuffer2[i] = load(B + (bid)*N*K + (iN)*K + (iKB+8));
      } else {
        bBuffer2[i] = 0.f;
      }
    }
  }
}

__device__ void load_ab_tn(
  const float* A,
  const float* B,
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4],
  int bid, int gStartx, int gStarty, int gStartk,
  int M, int N, int K
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  int iKA = gStartk + wy;
  int iKB = gStartk + wy;
  #pragma unroll
  for (int i=0; i<4; i++){
    int iM = gStarty + wx + i*32;
    int iN = gStartx + wx + i*32;
    if (likely(iM < M)){
      if (likely(iKA < K)){
        aBuffer1[i] = load(A + (bid)*M*K + (iKA)*M + (iM));
      } else {
        aBuffer1[i] = 0.f;
      }
      if (likely(iKA+8 < K)){
        aBuffer2[i] = load(A + (bid)*M*K + (iKA+8)*M + (iM));
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

__device__ void buffer2smem_nn(
  _VOLATILE_ float aSM1[8][128+4],
  _VOLATILE_ float aSM2[8][128+4],
  _VOLATILE_ float bSM1[8][128+4],
  _VOLATILE_ float bSM2[8][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM1[dx][dy+i*32] = aBuffer1[i];
    bSM1[wy][wx+i*32+i] = bBuffer1[i];
    aSM2[dx][dy+i*32] = aBuffer2[i];
    bSM2[wy][wx+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_tt(
  _VOLATILE_ float aSM1[8][128+4],
  _VOLATILE_ float aSM2[8][128+4],
  _VOLATILE_ float bSM1[8][128+4],
  _VOLATILE_ float bSM2[8][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM1[wy][wx+i*32] = aBuffer1[i];
    aSM2[wy][wx+i*32] = aBuffer2[i];
    bSM1[dx][dy+i*32+i] = bBuffer1[i];
    bSM2[dx][dy+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_nt(
  _VOLATILE_ float aSM1[8][128+4],
  _VOLATILE_ float aSM2[8][128+4],
  _VOLATILE_ float bSM1[8][128+4],
  _VOLATILE_ float bSM2[8][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM1[dx][dy+i*32] = aBuffer1[i];
    aSM2[dx][dy+i*32] = aBuffer2[i];
    bSM1[dx][dy+i*32+i] = bBuffer1[i];
    bSM2[dx][dy+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_tn(
  _VOLATILE_ float aSM1[8][128+4],
  _VOLATILE_ float aSM2[8][128+4],
  _VOLATILE_ float bSM1[8][128+4],
  _VOLATILE_ float bSM2[8][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM1[wy][wx+i*32] = aBuffer1[i];
    aSM2[wy][wx+i*32] = aBuffer2[i];
    bSM1[wy][wx+i*32+i] = bBuffer1[i];
    bSM2[wy][wx+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_16_nn(
  _VOLATILE_ float aSM[16][128+4],
  _VOLATILE_ float bSM[16][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM[dx][dy+i*32] = aBuffer1[i];
    aSM[dx+8][dy+i*32] = aBuffer2[i];
    bSM[wy][wx+i*32+i] = bBuffer1[i];
    bSM[wy+8][wx+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_16_tt(
  _VOLATILE_ float aSM[16][128+4],
  _VOLATILE_ float bSM[16][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM[wy][wx+i*32] = aBuffer1[i];
    aSM[wy+8][wx+i*32] = aBuffer2[i];
    bSM[dx][dy+i*32+i] = bBuffer1[i];
    bSM[dx+8][dy+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_16_nt(
  _VOLATILE_ float aSM[16][128+4],
  _VOLATILE_ float bSM[16][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM[dx][dy+i*32] = aBuffer1[i];
    aSM[dx+8][dy+i*32] = aBuffer2[i];
    bSM[dx][dy+i*32+i] = bBuffer1[i];
    bSM[dx+8][dy+i*32+i] = bBuffer2[i];
  }
}

__device__ void buffer2smem_16_tn(
  _VOLATILE_ float aSM[16][128+4],
  _VOLATILE_ float bSM[16][128+4],
  float aBuffer1[4],
  float aBuffer2[4],
  float bBuffer1[4],
  float bBuffer2[4]
){
  int tid = threadIdx.x;
  int wx = tid % 32;
  int wy = tid / 32;
  int dx = tid % 8;
  int dy = tid / 8;
  #pragma unroll
  for (int i=0; i<4; i++){
    // Store buffered tiles into shared memory
    aSM[wy][wx+i*32] = aBuffer1[i];
    aSM[wy+8][wx+i*32] = aBuffer2[i];
    bSM[wy][wx+i*32+i] = bBuffer1[i];
    bSM[wy+8][wx+i*32+i] = bBuffer2[i];
  }
}