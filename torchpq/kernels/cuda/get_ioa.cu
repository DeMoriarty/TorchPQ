typedef long long ll_t;

extern "C"
__global__ void get_ioa(
  const ll_t* __restrict__ labels,
  const ll_t* __restrict__ unique_labels,
  ll_t* __restrict__ ioa,
  int n_labels, int n_unique
) {
  int tid = threadIdx.x; // thread ID
  int uid = blockIdx.x * _TPB_ + tid; // unique label id

  ll_t ulabel = -1;
  int counter = 0;

  //ulabel = unique_labels[uid];
  if (uid < n_unique){
    ulabel = unique_labels[uid];
  } else {
    //ulabel = -1;
  }

  __shared__ volatile ll_t smem[_TPB_];
  int n_iter = (n_labels + _TPB_ - 1) / _TPB_;
  for (int i=0; i<n_iter; i++){
    int nid = i * _TPB_ + tid;
    if (nid < n_labels){
      smem[tid] = labels[nid];
    }
    __syncthreads();

#pragma unroll
    for (int j=0; j<_TPB_; j++){
      ll_t label = smem[j];
      if ( label == ulabel && (i * _TPB_ + j) < n_labels){
        ioa[i * _TPB_ + j] = ll_t(counter);
        counter ++;
      }
    }
    __syncthreads();
  }
}