typedef long long ll_t;

extern "C"
__global__ void get_div_of_address(
  const ll_t* __restrict__ address,
  const ll_t* __restrict__ div_start,
  const ll_t* __restrict__ div_end,
  ll_t* __restrict__ divs,
  int N, int K
) {
  int tid = threadIdx.x; // thread ID
  int bid = blockIdx.x; // block ID
  int div_id_start = bid * _TPB_ * _TA_ + tid * _TA_; //

  ll_t start_adr[_TA_];
  ll_t end_adr[_TA_];

#pragma unroll
  for (int i=0; i<_TA_; i++){
    if (div_id_start + i < K){
      start_adr[i] = div_start[div_id_start + i];
      end_adr[i] = div_end[div_id_start + i];
    }
  }
  __shared__ volatile ll_t smem[_TPB_];
  __shared__ volatile ll_t results[_TPB_];

  int n_iter = (N + _TPB_ - 1) / _TPB_;
  for (int i=0; i<n_iter; i++){
    int nid = i * _TPB_ + tid;

    if (nid < N){
      smem[tid] = address[nid];
    }
    results[tid] = -1;
    __syncthreads();

#pragma unroll
    for (int j=0; j<_TPB_; j++){
      ll_t adr = smem[j];
#pragma unroll
      for (int k=0; k<_TA_; k++){
        if (div_id_start + k < K){
          if (adr >= start_adr[k] && adr < end_adr[k]){
            results[j] = (ll_t) div_id_start + k;
          }
        }
      }
    }
    __syncthreads();

    if (nid < N){
      if (results[tid] >= 0)
        divs[nid] = results[tid];
    }
    __syncthreads();
  }
}