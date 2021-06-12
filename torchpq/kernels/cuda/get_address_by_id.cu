typedef long long ll_t;

extern "C"
__global__ void get_address_of_id(
  const ll_t* __restrict__ address2id,
  const ll_t* __restrict__ ids,
  ll_t* __restrict__ address,
  int n_data, int n_ids
) {
  int tid = threadIdx.x; // thread ID
  int qid = blockIdx.x * _TPB_ + tid;

  __shared__  volatile ll_t smem[_TPB_];

  ll_t id = -1;
  if (qid < n_ids){
    id = ids[qid];
  }

  int n_iter = (n_data + _TPB_ - 1) / _TPB_;
  for (int i=0; i<n_iter; i++){
    int nid = i * _TPB_ + tid;
    if (nid < n_data){
      smem[tid] = address2id[nid];
    }
    __syncthreads();
    if (qid < n_ids){
#pragma unroll
      for (int j=0; j<_TPB_; j++){
        if (i * _TPB_ + j < n_data){
          ll_t candidate = smem[j];
          if (candidate == id){
            address[qid] = (ll_t) (i * _TPB_ + j);
            break;
          }
        }
      }
    }
    __syncthreads();
  }
}