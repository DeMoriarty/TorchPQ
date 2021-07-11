typedef unsigned char uint8_t;
typedef long long int64_t;

extern "C"
__global__ void compute_centroids(
  const float* __restrict__ Data,
  const int64_t* __restrict__ Label,
  float* __restrict__ C,
  int M, int N, int E, int K
) {
  int tid = threadIdx.x; // thread ID
  int wid = tid / 32;
  int wtid = tid % 32;
  int nWarp = blockDim.x / 32;

  int sid = blockIdx.x; // subquantizer ID

  int kStart = blockIdx.y * _DK_;
  int kEnd = kStart + _DK_;
  if (kEnd > K)
    kEnd = K;
  int eStart = blockIdx.z * _DE_;
  int eEnd = eStart + _DK_;
  if (eEnd > E)
    eEnd = E;


  extern __shared__ volatile float  sh[]; // [(_DE_ + 1) * _DK_]
  // volatile int* counts = (volatile int*) &sh[_DE_ * _DK_];

  const int nIters = _NITERS_;
#pragma unroll
  for (int i = 0; i < nIters; i++){
    int ckid = i * _TPB_ + tid;
    if (ckid < _DK_){
      // counts[ckid] = 0;
      sh[_DK_*_DE_ + ckid] = 0;
#pragma unroll
      for (int j = 0; j < _DE_; j++){
        sh[(j) * _DK_ + (ckid) ] = 0.f;
      }
    }
  }
  __syncthreads();

  for (int iN = 0; iN < N / _TPB_ + int(N % _TPB_ != 0); iN++){
    const int nid = iN * _TPB_ + tid;

    if (nid >= N)
      break;
#pragma unroll
    for (int iE = 0; iE < _DE_; iE++){
      // load from Data [m, e, n]
      float val = Data[(sid)*E*N + (eStart + iE)*N + (nid)];

      // load from label
      int lab = (int) Label[(sid)*N + (nid)];

      // if label in range kStart : kEnd
      if (lab >= kStart && lab < kEnd){
        atomicAdd((float*) &sh[(iE) *_DK_ + (lab - kStart)], val);
        atomicAdd((float*) &sh[_DK_*_DE_ + (lab - kStart)], 1);
      }
    }
  }
  __syncthreads();
  // load from sh, divide by count, write to C
#pragma unroll
  for (int i = 0; i < nIters; i++){
    int ckid = i * _TPB_ + tid;
    if (ckid < _DK_ && ckid + kStart < K){
      //float count = (float) counts[ckid];
      float count = sh[_DK_*_DE_+ckid];
#pragma unroll
      for (int j = 0; j < _DE_; j++){
        float sum = sh[(j) * _DK_ + (ckid)];
        C[(sid)*E*K + (eStart + j)*K + (kStart + ckid)] = count == 0 ? 0.f : sum / count;
        // atomicAdd(&C[0 + (eStart)*K + (kStart + tid)], count == 0 ? -1.f : 99.f);
      }
    }
  }
}