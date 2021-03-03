typedef long long ll_t;

extern "C"
__global__ void get_div_of_address(
  const ll_t* __restrict__ address,
  const ll_t* __restrict__ divStart,
  const ll_t* __restrict__ divEnd,
  ll_t* __restrict__ divs,
  int nAddress, int nDivs
) {
  int tid = threadIdx.x; // thread ID
  int bid = blockIdx.x; // block ID
  int aStartBlock = bid * _TPB_ * _TA_;
  // int aStartThread = aStartBlock + tid * _TA_;

  extern __shared__ ll_t smem[];  //[_TPB_ * _TA_]

  ll_t threadAddress[_TA_];
#pragma unroll
  for (int i=0; i<_TA_; i++){
    int aid = aStartBlock + i * _TPB_ + tid;
    if (aid < nAddress){
      threadAddress[i] = address[aid];
    } else {
      threadAddress[i] = -3;
    }
  }

#pragma unroll
  for (int i=0; i<_TA_; i++){
    int idx = i * _TPB_ + tid;
    smem[idx] = threadAddress[i];

  }
  __syncthreads();
  ll_t threadMax = -1;
#pragma unroll
  for (int i=0; i<_TA_; i++){
    int idx = tid * _TA_ + i;
    threadAddress[i] = smem[idx];
    threadMax = max(threadMax, threadAddress[i]);
  }
  ll_t threadMin = threadAddress[0];

  int nIters = __float2int_rn(__log2f(float(nDivs))) + 1;
  
  int leftPivotRange[2] = {0, nDivs};
  int leftPivot = 0;
  for (int i=0; i<nIters; i++){
    leftPivot = (leftPivotRange[0] + leftPivotRange[1]) / 2;
    ll_t leftDivStart = divStart[leftPivot];
    ll_t leftDivEnd = divEnd[leftPivot];
    if (leftDivStart <= threadMin && leftDivEnd > threadMin){
      break;
    } else if (leftDivStart > threadMin) {
      leftPivotRange[1] = leftPivot;
    } else if (leftDivEnd <= threadMin) {
      leftPivotRange[0] = leftPivot;
    }
  } // end for i

  int rightPivotRange[2] = {0, nDivs};
  int rightPivot = nDivs - 1;
  for (int i=0; i<nIters; i++){
    rightPivot = (rightPivotRange[0] + rightPivotRange[1]) / 2;
    ll_t rightDivStart = divStart[rightPivot];
    ll_t rightDivEnd = divEnd[rightPivot];
    if (rightDivStart <= threadMax && rightDivEnd > threadMax){
      break;
    } else if (rightDivStart > threadMax) {
      rightPivotRange[1] = rightPivot;
    } else if (rightDivEnd <= threadMax) {
      rightPivotRange[0] = rightPivot;
    }
  } // end for i

  for (int i=leftPivot; i<rightPivot + 1; i++){ // is +1 necessary?
    ll_t cDivStart = divStart[i];
    ll_t cDivEnd = divEnd[i];
#pragma unroll
    for (int j=0; j<_TA_; j++){
      int aid = aStartBlock + tid * _TA_ + j;
      if (aid < nAddress){
        ll_t adr = threadAddress[j];
        if (adr >= cDivStart && adr < cDivEnd){
          divs[aid] = i;
        }
      }
    } // end for j
  } // end for i
}