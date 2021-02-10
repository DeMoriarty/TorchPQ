typedef unsigned char uint8_t;

extern "C"
__global__ void pq_decode(
  const float* __restrict__ codebook,
  const uint8_t* __restrict__ code,
  float* __restrict__ result,
  int M, int D, int N
) {
  const int tid = threadIdx.x; // thread ID
  const int mStart = blockIdx.x * _TM_;
  const int dStart = blockIdx.y * _TD_;

  extern __shared__ volatile float smem[]; // _TM_ * _TD_ * 256

  // load codebook into smem
#pragma unroll
  for (int m=0; m<_TM_; m++){
#pragma unroll
    for (int d=0; d<_TD_; d++){
      int mid = mStart + m;
      int did = dStart + d;
      if (mid < M && did < D){
        smem[(m)*_TD_*256 + (d)*256 + tid] = codebook[(mid)*D*256 + (did)*256 + tid];
      }
    }
  }
  __syncthreads();

  int nIter = (N + _TPB_ - 1) / _TPB_;
  for (int i=0; i<nIter; i++){
    int nid = i * _TPB_ + tid;
    if (nid < N){
#pragma unroll
      for (int m=0; m<_TM_; m++){
        int mid = mStart + m;
        int index = code[(mid)*N + (nid)];
#pragma unroll
        for (int d=0; d<_TD_; d++){
          int did = dStart + d;
          if (mid < M && did < D){
            float value = smem[(m)*_TD_*256 + (d)*256 + index];
            result[(mid)*D*N + (did)*N + nid] = value;
          }
        }
      }
    }
  }
  __syncthreads();
}