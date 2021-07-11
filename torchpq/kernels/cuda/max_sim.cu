typedef long long ll_t;
typedef unsigned long long ull_t;

typedef struct __builtin_align__(32) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
} _float8;

typedef union {
  _float8 f8;
  float val[8];
} float8;

__device__ __forceinline__ float atomicMax(float *address, float val)
{
  int ret = __float_as_int(*address);
  while(val > __int_as_float(ret))
  {
    int old = ret;
    if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        break;
  }
  return __int_as_float(ret);
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

__device__ void SM2Cache(
  float cache[8][4],
  volatile float SM[8][128+4],
  int vy, int p
) {
#pragma unroll
  for (int ki=0; ki<8; ki++){
#pragma unroll
    for (int mi=0; mi<4; mi++){
      cache[ki][mi] = SM[ki][8*vy + 4*p + mi];
    }
  }
}

__device__ void thread_matmul(
  float aCache[8][4],
  volatile float bSM[8][128+4],
  float8 cCache[8],
  int vx, int p
) {
#pragma unroll
  for (int ki=0; ki<8; ki++){
#pragma unroll
    for (int ni=0; ni<8; ni++){
      // float b = bSM[ki][(8*vx)/32 + 8*vx + ni];
      float b = bSM[ki][ vx/4 + 8*vx + ni];
#pragma unroll
      for (int mi=0; mi<4; mi++){
        float a = aCache[ki][mi];
        cCache[mi + 4*p].val[ni] = fmaf(a, b, cCache[mi + 4*p].val[ni]);
      }
    }
  }
}

// negative squared euclidean distance
__device__ void thread_nseuclidean(
  float aCache[8][4],
  volatile float bSM[8][128+4],
  float8 cCache[8],
  int vx, int p
) {
#pragma unroll
  for (int ki=0; ki<8; ki++){
#pragma unroll
    for (int ni=0; ni<8; ni++){
      // float b = bSM[ki][(8*vx)/32 + 8*vx + ni];
      float b = bSM[ki][ vx/4 + 8*vx + ni];
#pragma unroll
      for (int mi=0; mi<4; mi++){
        float a = aCache[ki][mi];
        float dif = a - b;
        cCache[mi + 4*p].val[ni] = fmaf(- dif, dif, cCache[mi + 4*p].val[ni]);
      }
    }
  }
}

// negative manhattan distance
__device__ void thread_nmanhattan(
  float aCache[8][4],
  volatile float bSM[8][128+4],
  float8 cCache[8],
  int vx, int p
) {
#pragma unroll
  for (int ki=0; ki<8; ki++){
#pragma unroll
    for (int ni=0; ni<8; ni++){
      // float b = bSM[ki][(8*vx)/32 + 8*vx + ni];
      float b = bSM[ki][ vx/4 + 8*vx + ni];
#pragma unroll
      for (int mi=0; mi<4; mi++){
        float a = aCache[ki][mi];
        cCache[mi + 4*p].val[ni] -= fabsf(a - b);
      }
    }
  }
}

__device__ void reduce_dim_1(
  float8 cCache[8],
  float* C,
  ll_t* D,
  int gStartx, int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
#pragma unroll
  for (int i=0; i<8; i++){
    int iN = gStartx + vx*8 + i;
    float val = -INFINITY;
    ll_t ind = 0;
#pragma unroll
    for (int j=0; j<8; j++){
      int iM = gStarty + vy*8 + j;
      if (iM < _M_){
        val = fmaxf(val, cCache[j].val[i]);
        ind = val == cCache[j].val[i] ? iM : ind;
      }
    }
    if (iN < N){
      atomicMax(&C[(bid) * _N_ + iN], val);
      if (C[(bid) * _N_ + iN] <= val){
        D[(bid) * _N_ + iN] = ind;
      }
    }
  }
}

__device__ void reduce_dim_2(
  float8 cCache[8],
  float* C,
  ll_t* D,
  int gStartx, int gStarty,
  int vx, int vy, int bid,
  int M, int N
) {
#pragma unroll
  for (int i=0; i<8; i++){
    int iM = gStarty + vy*8 + i;
    float val = -INFINITY;
    ll_t ind = 0;
#pragma unroll
    for (int j=0; j<8; j++){
      int iN = gStartx + vx*8 + j;
      if (iN < _N_){
        val = fmaxf(val, cCache[i].val[j]);
        ind = val == cCache[i].val[j] ? iN : ind;
      }
    }
    if (iM < M){
      atomicMax(&C[(bid) * _M_ + iM], val);
      if (C[(bid) * _M_ + iM] <= val){
        D[(bid) * _M_ + iM] = ind;
      }
    }
  }
}

extern "C"
__global__ void max_sim_tn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  ll_t* __restrict__ D,
  int M, int N, int K, int DIM
){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gStartx = blockIdx.y * 128;
  int gStarty = blockIdx.z * 128;

  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ volatile float aSM[8][128+4];
  __shared__ volatile float bSM[8][128+4];
  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  int nIt = (_K_ + 8 - 1) / 8;
  float init_value = 0.f;
#pragma unroll
  for (int i=0; i<4; i++){

    int iM = gStarty + wx + i*32;
    int iN = gStartx + wx + i*32;
    if (wy < _K_){
      if (iM < _M_)
        aBuffer1[i] = A[(bid)*_M_*_K_ + (wy)*_M_ + (iM)];
      if (iN < _N_)
        bBuffer1[i] = B[(bid)*_N_*_K_ + (wy)*_N_ + (gStartx + wx + i*32)];
    } else {
      aBuffer1[i] = 0.f;
      bBuffer1[i] = 0.f;
    }
  }
#pragma unroll
  for (int itr=0; itr<nIt; itr++){
    
    int gStartk = itr * 8;
    int iK = gStartk + 8 + wy;
    int is_odd = itr & 1;
    if (is_odd == 0){
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + wx;
          int iN = gStartx + i*32 + wx;
          
          if (iK < _K_){
            if (iM < _M_)
              aBuffer2[i] = A[(bid)*_M_*_K_ + (iK)*_M_ + (iM)];
            if (iN < _N_)
              bBuffer2[i] = B[(bid)*_N_*_K_ + (iK)*_N_ + (iN)];
          } else {
            aBuffer2[i] = 0.f;
            bBuffer2[i] = 0.f;
          }
        }
        aSM[wy][wx+i*32] = aBuffer1[i];
        bSM[wy][wx+i*32+i] = bBuffer1[i];
      }
    } else {
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + wx;
          int iN = gStartx + i*32 + wx;
          if (iK < _K_){
            if (iM < _M_)
              aBuffer1[i] = A[(bid)*_M_*_K_ + (iK)*_M_ + (iM)];
            if (iN < N)
              bBuffer1[i] = B[(bid)*_N_*_K_ + (iK)*_N_ + (iN)];
          } else {
            aBuffer1[i] = 0.f;
            bBuffer1[i] = 0.f;
          }
        }
        aSM[wy][wx+i*32] = aBuffer2[i];
        bSM[wy][wx+i*32+i] = bBuffer2[i];
      }
    }
    __syncthreads();

    float aCache[8][4];

#pragma unroll
    for (int p=0; p<2; p++){
      SM2Cache(aCache, aSM, vy, p);
      // thread_matmul(aCache, bSM, cCache, vx, p);
      _DISTFN_(aCache, bSM, cCache, vx, p);
    }
    __syncthreads();
  }

  if (DIM == 1){
    reduce_dim_1(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );

  } else if (DIM == 2){
    reduce_dim_2(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );
  }
}

extern "C"
__global__ void max_sim_nt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  ll_t* __restrict__ D,
  int M, int N, int K, int DIM
){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gStartx = blockIdx.y * 128;
  int gStarty = blockIdx.z * 128;

  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ volatile float aSM[8][128+4];
  __shared__ volatile float bSM[8][128+4];
  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  int nIt = (_K_ + 8 - 1) / 8;
  float init_value = 0.f;
#pragma unroll
  for (int i=0; i<4; i++){

    int iM = gStarty + dy + i*32;
    int iN = gStartx + dy + i*32;
    if (dx < _K_){
      if (iM < _M_)
        aBuffer1[i] = A[(bid)*_M_*_K_ + (iM)*_K_ + (dx)];
      if (iN < _N_)
        bBuffer1[i] = B[(bid)*_N_*_K_ + (iN)*_K_ + (dx)];
    } else {
      aBuffer1[i] = 0.f;
      bBuffer1[i] = 0.f;
    }
  }
#pragma unroll
  for (int itr=0; itr<nIt; itr++){
    
    int gStartk = itr * 8;
    int iK = gStartk + 8 + dx;
    int is_odd = itr & 1;
    if (is_odd == 0){
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + dy;
          int iN = gStartx + i*32 + dy;
          
          if (iK < _K_){
            if (iM < _M_)
              aBuffer2[i] = A[(bid)*_M_*_K_ + (iM)*_K_ + (iK)];
            if (iN < _N_)
              bBuffer2[i] = B[(bid)*_N_*_K_ + (iN)*_K_ + (iK)];
          } else {
            aBuffer2[i] = 0.f;
            bBuffer2[i] = 0.f;
          }
        }
        aSM[dx][dy+i*32] = aBuffer1[i];
        bSM[dx][dy+i*32+i] = bBuffer1[i];
      }
    } else {
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + dy;
          int iN = gStartx + i*32 + dy;
          if (iK < _K_){
            if (iM < _M_)
              aBuffer1[i] = A[(bid)*_M_*_K_ + (iM)*_K_ + (iK)];
            if (iN < N)
              bBuffer1[i] = B[(bid)*_N_*_K_ + (iN)*_K_ + (iK)];
          } else {
            aBuffer1[i] = 0.f;
            bBuffer1[i] = 0.f;
          }
        }
        aSM[dx][dy+i*32] = aBuffer2[i];
        bSM[dx][dy+i*32+i] = bBuffer2[i];
      }
    }
    __syncthreads();

    float aCache[8][4];

#pragma unroll
    for (int p=0; p<2; p++){
      SM2Cache(aCache, aSM, vy, p);
      _DISTFN_(aCache, bSM, cCache, vx, p);
    }
    __syncthreads();
  }

  if (DIM == 1){
    reduce_dim_1(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );

  } else if (DIM == 2){
    reduce_dim_2(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );
  }
}

extern "C"
__global__ void max_sim_nn(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  ll_t* __restrict__ D,
  int M, int N, int K, int DIM
){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gStartx = blockIdx.y * 128;
  int gStarty = blockIdx.z * 128;

  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ volatile float aSM[8][128+4];
  __shared__ volatile float bSM[8][128+4];
  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  int nIt = (_K_ + 8 - 1) / 8;
  float init_value = 0.f;
#pragma unroll
  for (int i=0; i<4; i++){

    int iM = gStarty + dy + i*32;
    int iN = gStartx + wx + i*32;
    if (iM < _M_){
      if (dx < _K_){
        aBuffer1[i] = A[(bid)*_M_*_K_ + (iM)*_K_ + (dx)];
      } else {
        aBuffer1[i] = 0.f;
      }
    }
    if (iN < N){
      if (wy < _K_){
        bBuffer1[i] = B[(bid)*_N_*_K_ + (wy)*_N_ + (iN)];
      } else {
        bBuffer1[i] = 0.f;
      }
    }

  }
#pragma unroll
  for (int itr=0; itr<nIt; itr++){
    
    int gStartk = itr * 8;
    int iKA = gStartk + 8 + dx;
    int iKB = gStartk + 8 + wy;
    int is_odd = itr & 1;
    if (is_odd == 0){
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + dy;
          int iN = gStartx + i*32 + wx;
          
          if (iKA < _K_){
            if (iM < _M_){
              aBuffer2[i] = A[(bid)*_M_*_K_ + (iM)*_K_ + (iKA)];
            }
          } else {
            aBuffer2[i] = 0.f;
          }

          if (iKB < _K_){
            if (iN < _N_){
              bBuffer2[i] = B[(bid)*_N_*_K_ + (iKB)*_N_ + (iN)];
            }
          } else {
            bBuffer2[i] = 0.f;
          }
        }
        aSM[dx][dy+i*32] = aBuffer1[i];
        bSM[wy][wx+i*32+i] = bBuffer1[i];
      }
    } else {
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + dy;
          int iN = gStartx + i*32 + wx;
          if (iKA < _K_){
            if (iM < _M_){
              aBuffer1[i] = A[(bid)*_M_*_K_ + (iM)*_K_ + (iKA)];
            }
          } else {
            aBuffer1[i] = 0.f;
          }

          if (iKB < _K_){
            if (iN < _N_){
              bBuffer1[i] = B[(bid)*_N_*_K_ + (iKB)*_N_ + (iN)];
            }
          } else {
            bBuffer1[i] = 0.f;
          }
        }
        aSM[dx][dy+i*32] = aBuffer2[i];
        bSM[wy][wx+i*32+i] = bBuffer2[i];
      }
    }
    __syncthreads();

    float aCache[8][4];

#pragma unroll
    for (int p=0; p<2; p++){
      SM2Cache(aCache, aSM, vy, p);
      _DISTFN_(aCache, bSM,cCache,  vx, p);
    }
    __syncthreads();
  }

  if (DIM == 1){
    reduce_dim_1(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );

  } else if (DIM == 2){
    reduce_dim_2(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );
  }
}

extern "C"
__global__ void max_sim_tt(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  ll_t* __restrict__ D,
  int M, int N, int K, int DIM
){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gStartx = blockIdx.y * 128;
  int gStarty = blockIdx.z * 128;

  int vx = tid % 16;
  int vy = tid / 16;
  int wx = tid % 32; // thread idx in warp
  int wy = tid / 32; // warp id
  int dx = tid % 8;
  int dy = tid / 8;

  __shared__ volatile float aSM[8][128+4];
  __shared__ volatile float bSM[8][128+4];
  float aBuffer1[4];
  float bBuffer1[4];
  float aBuffer2[4];
  float bBuffer2[4];

  float8 cCache[8];
  init_cCache(cCache);

  int nIt = (_K_ + 8 - 1) / 8;
  float init_value = 0.f;
#pragma unroll
  for (int i=0; i<4; i++){

    int iM = gStarty + wx + i*32;
    int iN = gStartx + dy + i*32;
    if (iM < _M_){
      if (wy < _K_){
        aBuffer1[i] = A[(bid)*_M_*_K_ + (wy)*_M_ + (iM)];
      } else {
        aBuffer1[i] = 0.f;
      }
    }
    if (iN < _N_){
      if (dx < _K_){
        bBuffer1[i] = B[(bid)*_N_*_K_ + (iN)*_K_ + (dx)];
      } else {
        bBuffer1[i] = 0.f;
      }
    }
  }
#pragma unroll
  for (int itr=0; itr<nIt; itr++){
    
    int gStartk = itr * 8;
    int iKA = gStartk + 8 + wy;
    int iKB = gStartk + 8 + dx;
    int is_odd = itr & 1;
    if (is_odd == 0){
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + wx;
          int iN = gStartx + i*32 + dy;
          
          if (iKA < _K_){
            if (iM < _M_){
              aBuffer2[i] = A[(bid)*_M_*_K_ + (iKA)*_M_ + (iM)];
            }
          } else {
            aBuffer2[i] = 0.f;
          }

          if (iKB < _K_){
            if (iN < _N_){
              bBuffer2[i] = B[(bid)*_N_*_K_ + (iN)*_K_ + (iKB)];
            }
          } else {
            bBuffer2[i] = 0.f;
          }
        }
        aSM[wy][wx+i*32] = aBuffer1[i];
        bSM[dx][dy+i*32+i] = bBuffer1[i];
      }
    } else {
#pragma unroll
      for (int i=0; i<4; i++){
        if (itr < nIt - 1){
          int iM = gStarty + i*32 + wx;
          int iN = gStartx + i*32 + dy;
          if (iKA < _K_){
            if (iM < _M_){
              aBuffer1[i] = A[(bid)*_M_*_K_ + (iKA)*_M_ + (iM)];
            }
          } else {
            aBuffer1[i] = 0.f;
          }

          if (iKB < _K_){
            if (iN < _N_){
              bBuffer1[i] = B[(bid)*_N_*_K_ + (iN)*_K_ + (iKB)];
            }
          } else {
            bBuffer1[i] = 0.f;
          }
        }
        aSM[wy][wx+i*32] = aBuffer2[i];
        bSM[dx][dy+i*32+i] = bBuffer2[i];
      }
    }
    __syncthreads();

    float aCache[8][4];

#pragma unroll
    for (int p=0; p<2; p++){
      SM2Cache(aCache, aSM, vy, p);
      _DISTFN_(aCache, bSM, cCache, vx, p);
    }
    __syncthreads();
  }

  if (DIM == 1){
    reduce_dim_1(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );

  } else if (DIM == 2){
    reduce_dim_2(
      cCache, C, D,
      gStartx, gStarty,
      vx, vy, bid, M, N
    );
  }
}