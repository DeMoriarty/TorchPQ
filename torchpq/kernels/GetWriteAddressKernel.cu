typedef long long ll_t;

extern "C"
__global__ void get_write_address(
  const ll_t* __restrict__ empty_adr,
  const ll_t* __restrict__ div_of_empty_adr,
  const ll_t* __restrict__ labels,
  const ll_t* __restrict__ ioa,
  ll_t* __restrict__ write_adr,
  int n_empty, int n_labels
) {
  int tid = threadIdx.x; // thread ID
  int lid = blockIdx.x * _TPB_ + tid; // label ID

  ll_t label;
  ll_t write_at_count;
  ll_t counter = 0;
  if (lid < n_labels){
    label = labels[lid];
    write_at_count = ioa[lid];
  } else {
    label = -1;
    write_at_count = -1;
  }

  __shared__ volatile ll_t smem[_TPB_];
  int n_iter = (n_empty + _TPB_ - 1) / _TPB_;
  for (int i =0; i<n_iter; i++){
    int nid = i * _TPB_ + tid;
    if (nid < n_empty){
      smem[tid] = div_of_empty_adr[nid];
    }
    __syncthreads();
    if (counter <= write_at_count){
      for (int j=0; j<_TPB_; j++){
        ll_t div = smem[j];
        if (div == label && i*_TPB_ + j < n_empty){
          if (counter == write_at_count){
            ll_t adr = empty_adr[i*_TPB_ + j];
            write_adr[lid] = adr;
          }
          counter ++;
        }
      }
    }
    __syncthreads();
  }
}