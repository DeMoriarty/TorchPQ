typedef unsigned char uint8_t;

struct __device_builtin__ __align__(_NCS_) uint8n
{
    uint8_t _VARNAMES_;
};

extern "C"
__global__ void compute_product(
  const uint8_t* __restrict__ A,
  const float* __restrict__ B,
  const char* __restrict__ isEmpty,
  const int* __restrict__ divStart,
  const int* __restrict__ divSize,
  float* __restrict__ V,
  int* __restrict__ I,
  int N, int L, int O, int nProbe
) {
  const int tid = threadIdx.x; // thread ID
  const int qid = blockIdx.x; // query ID
  // const uint8n* A2 = reinterpret_cast<const uint8n*>( const_cast<uint8_t*>(A) )
  const uint8n* A2 = reinterpret_cast<const uint8n*>(A); // ?

  // Load precomputed distances
  extern __shared__ volatile float Bsh[];
#pragma unroll
  if (tid < 256){
    for (int i = 0; i < _M_; i++){
      int bz = i;
      int by = qid;
      int bx = tid;
      Bsh[i * _K_ + tid] = B[(bz * L * _K_) + (by * _K_) + (bx)];
    }
  }
  __syncthreads();
  // Load A and compute distance
  int iN = tid;
  int counter = tid;
  int start = 0; 
  int size = 0;
  int cDiv = -1;
  bool break_loop = false;
  while (iN < N){
    while ( (iN - start) >= size){
      cDiv ++;
      if (cDiv >= nProbe){
        break_loop = true;
        break;
      }
      int residual = iN - start - size;
      start = divStart[(qid) * nProbe + (cDiv)];
      iN = start + residual;
      size =  divSize[(qid) * nProbe + (cDiv)];
      if (iN >= N){
        break_loop = true;
        break;
      }
    }
    if (break_loop)
      break;

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < _M_ / _NCS_; i++){
      uint8n Avals = A2[(i * N) + (iN)];
_CODEBLOCK_
    }
    // write to V and I
    int isCurrentEmpty;
    isCurrentEmpty = isEmpty[iN];

    /*
    if (isCurrentEmpty == 0){
      V[(qid) * O + counter] = sum;
      I[(qid) * O + counter] = iN;
    } else {
      V[(qid) * O + counter] = -999999.f;
      I[(qid) * O + counter] = -1;
    }
    */
    
    if (counter < O){
      V[(qid) * O + counter] = isCurrentEmpty == 0 ? sum : -999999.f;
      I[(qid) * O + counter] = isCurrentEmpty == 0 ? iN : -1;
      // atomicAdd(V + (qid) * O + counter, isCurrentEmpty == 0 ? sum : -99999.f);
    }   
    iN += _TPB_;
    counter += _TPB_;
  }
}